#!/usr/bin/env python3
"""
Forward-Hard (Hierarchical) Diffusion — Minimal PyTorch Prototype

What this script does
---------------------
- Trains a score network with **denoising score matching** on CIFAR-10 (32×32).
- The corruption kernel is **block-mean biased + anisotropic Gaussian**:
    x' = (1-a) x + a P_s x + sigma_R * R_s z + sigma_P * P_s z + sigma_iso * z2
  where P_s is the block-mean projection on an s×s grid, and R_s = I - P_s.
- The network learns the **score** of q(x'|x); training target is
    ∇_{x'} log q(x'|x) = - Σ^{-1} (x' - [(1-a)x + a P_s x])
  with Σ = (sigma_R^2) R_s + (sigma_P^2) P_s + (sigma_iso^2) I.
- Sampling uses an **anisotropic Langevin** predictor with a multiscale schedule
  on s and sigmas (coarse → fine).

This is a research prototype meant to be readable and hackable. It will train
and produce recognizable CIFAR-ish samples in a short run on a single GPU,
though not SOTA quality.

Usage
-----
python forward_hard_diffusion_pytorch.py --epochs 3 --batch_size 128 --samples_every 1

Dependencies
------------
- torch, torchvision, tqdm

Notes
-----
- For speed, the UNet is intentionally small.
- Images are scaled to [-1, 1].
- Saved outputs go to ./runs/<timestamp>/
"""

import math
import os
import time
import copy
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

# ---------------------------- Utils: projections ---------------------------- #

def block_project(x: torch.Tensor, s: int) -> torch.Tensor:
    """Project x onto piecewise-constant fields on an s×s grid via average pooling.
    x: (B, C, H, W), H and W divisible by s.
    """
    if s == 1:
        return x
    B, C, H, W = x.shape
    assert H % s == 0 and W % s == 0, f"Image size must be divisible by s (got H={H}, W={W}, s={s})"
    y = F.avg_pool2d(x, kernel_size=s, stride=s)
    y_up = F.interpolate(y, scale_factor=s, mode="nearest")
    return y_up


def P_apply(x: torch.Tensor, s: int) -> torch.Tensor:
    return block_project(x, s)


def R_apply(x: torch.Tensor, s: int) -> torch.Tensor:
    return x - block_project(x, s)


def Sigma_inv_apply(v: torch.Tensor, s: int, sigma_R: float, sigma_P: float, sigma_iso: float) -> torch.Tensor:
    """Apply Σ^{-1} where Σ = sigma_R^2 * R + sigma_P^2 * P + sigma_iso^2 * I.
    Using orthogonal projections R and P: Σ^{-1} v = v_R/(sig_R^2+iso^2) + v_P/(sig_P^2+iso^2)
    """
    Pv = P_apply(v, s)
    Rv = v - Pv
    return Rv / ((sigma_R ** 2) + (sigma_iso ** 2)) + Pv / ((sigma_P ** 2) + (sigma_iso ** 2))


def Sigma_apply(v: torch.Tensor, s: int, sigma_R: float, sigma_P: float, sigma_iso: float) -> torch.Tensor:
    """Apply Σ where Σ = sigma_R^2 * R + sigma_P^2 * P + sigma_iso^2 * I."""
    Pv = P_apply(v, s)
    Rv = v - Pv
    return Rv * ((sigma_R ** 2) + (sigma_iso ** 2)) + Pv * ((sigma_P ** 2) + (sigma_iso ** 2))


def forward_one_step(x: torch.Tensor, s: int, a: float, sigma_R: float, sigma_P: float, sigma_iso: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward corruption step from clean x.
    Returns (x_next, mean_m) where mean_m = (1-a) x + a P_s x.
    """
    with torch.no_grad():
        m = (1 - a) * x + a * P_apply(x, s)
        z = torch.randn_like(x)
        z2 = torch.randn_like(x)
        noise = sigma_R * R_apply(z, s) + sigma_P * P_apply(z, s) + sigma_iso * z2
        x_next = m + noise
    return x_next, m

# ------------------------------ Model (UNet) ------------------------------- #

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // (2 * x.shape[1])
        outs = []
        for i in range(x.shape[1]):
            freq = torch.arange(half, device=x.device, dtype=x.dtype)
            freq = 1.0 / (10.0 ** (freq / half))
            angle = x[:, i:i+1] * freq[None, :]
            outs += [torch.sin(2 * math.pi * angle), torch.cos(2 * math.pi * angle)]
        emb = torch.cat(outs, dim=1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        self.rb = ResBlock(in_ch, out_ch, emb_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.rb(x, emb)
        skip = h
        h = self.down(h)
        return h, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.rb = ResBlock(out_ch + skip_ch, out_ch, emb_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.rb(h, emb)
        return h


class SmallUNet(nn.Module):
    """A proper U-Net encoder–decoder with skip connections (32x32 friendly).
    Keeps the same class name so training code doesn't change.
    """
    def __init__(self, in_ch=3, base=64, emb_dim=256, cond_dim=8):
        super().__init__()
        # conditioning embedding
        self.sin_emb = SinusoidalEmbedding(dim=cond_dim * 16)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 16, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        # encoder
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = Down(base, base, emb_dim)       # 32 -> 16
        self.down2 = Down(base, base * 2, emb_dim)   # 16 -> 8
        self.down3 = Down(base * 2, base * 4, emb_dim)  # 8 -> 4

        # bottleneck
        self.mid1 = ResBlock(base * 4, base * 4, emb_dim)
        self.mid2 = ResBlock(base * 4, base * 4, emb_dim)

        # decoder
        self.up3 = Up(base * 4, base * 4, base * 2, emb_dim)  # 4 -> 8
        self.up2 = Up(base * 2, base * 2, base, emb_dim)      # 8 -> 16
        self.up1 = Up(base, base, base, emb_dim)              # 16 -> 32

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        emb = self.cond_mlp(self.sin_emb(cond_vec))

        # encoder
        h0 = self.in_conv(x)
        h1, s1 = self.down1(h0, emb)  # s1: 32x32
        h2, s2 = self.down2(h1, emb)  # s2: 16x16
        h3, s3 = self.down3(h2, emb)  # s3: 8x8

        # bottleneck
        h = self.mid1(h3, emb)
        h = self.mid2(h, emb)

        # decoder
        h = self.up3(h, s3, emb)
        h = self.up2(h, s2, emb)
        h = self.up1(h, s1, emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h  # predicts score field s_theta(x)

# ------------------------------ Schedules ---------------------------------- #

@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_root: str = "./runs"
    epochs: int = 3
    batch_size: int = 128
    lr: float = 2e-4
    num_workers: int = 4
    # Prefer Apple MPS > CUDA > CPU
    device: str = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    image_size: int = 32
    save_every: int = 1
    samples_every: int = 1
    seed: int = 42


# Multiscale grid sizes for CIFAR-10
S_LIST = [1, 4, 8, 16]  # 1 keeps identity; others impose block means


def sample_training_hyperparams(device: str) -> Dict[str, torch.Tensor]:
    """Sample a SINGLE set of hyperparams for the whole batch (to avoid label/corruption mismatch)."""
    a = torch.rand((), device=device) * 0.6 + 0.2
    sigma_P = torch.rand((), device=device) * 0.25 + 0.05
    sigma_R = sigma_P + torch.rand((), device=device) * 0.25 + 0.05
    sigma_iso = torch.rand((), device=device) * 0.05
    s = torch.tensor(S_LIST[torch.randint(0, len(S_LIST), (1,), device=device).item()], device=device, dtype=torch.float32)
    return {"a": a, "sigma_P": sigma_P, "sigma_R": sigma_R, "sigma_iso": sigma_iso, "s": s}


def build_cond_vector(hp: Dict[str, torch.Tensor], batch: Optional[int] = None) -> torch.Tensor:
    """Build conditioning vector.
    Accepts scalars (for whole-batch settings) or per-sample 1D tensors.
    If scalars are provided, they are expanded to (batch, 1)."""
    # infer batch size
    if batch is None:
        # try to infer from any tensor with numel()>1; otherwise default to 1
        sizes = [v.numel() for v in hp.values()]
        batch = max(sizes) if max(sizes) > 1 else 1

    def _expand(x):
        if x.dim() == 0:
            return x.view(1, 1).expand(batch, 1)
        elif x.dim() == 1:
            return x.view(-1, 1)
        else:
            return x

    a = _expand(hp["a"])
    sigma_P = _expand(hp["sigma_P"])
    sigma_R = _expand(hp["sigma_R"])
    sigma_iso = _expand(hp["sigma_iso"])
    s = _expand(hp["s"])

    cond = torch.cat([
        a,
        sigma_P,
        sigma_R,
        sigma_iso,
        torch.log2(s) / 5.0,
        1.0 / s,
        sigma_R / (sigma_P + 1e-6),
        a * sigma_R,
    ], dim=1)
    return cond

# ------------------------------- Training ---------------------------------- #

def ema_update(src: nn.Module, tgt: nn.Module, decay: float = 0.999):
    with torch.no_grad():
        ms = src.state_dict()
        mt = tgt.state_dict()
        for k in mt.keys():
            mt[k].mul_(decay).add_(ms[k], alpha=1.0 - decay)
        tgt.load_state_dict(mt)


def make_dataloader(cfg: TrainConfig) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1,1]
    ])
    ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=tfm)
    # pin_memory helps CUDA but not MPS/CPU
    pin = isinstance(cfg.device, str) and cfg.device.startswith("cuda")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_root, exist_ok=True)
    run_dir = os.path.join(cfg.out_root, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    device = cfg.device
    print(f"[info] Using device: {device}")
    loader = make_dataloader(cfg)
    net = SmallUNet(in_ch=3, base=64, emb_dim=192, cond_dim=8).to(device)
    net_ema = copy.deepcopy(net).to(device)
    for p in net_ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        ema_loss = None
        for (x0, _) in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            hp = sample_training_hyperparams(device)  # single set per batch
            cond = build_cond_vector(hp, B)

            # Forward one step: x1 = m + noise (shared hyperparams across batch)
            x1, m = forward_one_step(
                x0,
                s=int(hp["s"].item()),
                a=float(hp["a"].item()),
                sigma_R=float(hp["sigma_R"].item()),
                sigma_P=float(hp["sigma_P"].item()),
                sigma_iso=float(hp["sigma_iso"].item()),
            )

            # Target score: -Sigma^{-1}(x1 - m)
            target = -Sigma_inv_apply(
                x1 - m,
                s=int(hp["s"].item()),
                sigma_R=float(hp["sigma_R"].item()),
                sigma_P=float(hp["sigma_P"].item()),
                sigma_iso=float(hp["sigma_iso"].item()),
            )

            pred = net(x1, cond)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema_update(net, net_ema, decay=0.999)

            global_step += 1
            ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
            pbar.set_postfix(loss=f"{ema_loss:.4f}")

        # Save checkpoints
        torch.save({"model": net.state_dict(), "cfg": cfg.__dict__}, os.path.join(run_dir, f"ckpt_epoch_{epoch}.pt"))
        torch.save({"model": net_ema.state_dict(), "cfg": cfg.__dict__}, os.path.join(run_dir, f"ckpt_epoch_{epoch}_ema.pt"))

        if epoch % cfg.samples_every == 0:
            with torch.no_grad():
                net_ema.eval()
                imgs = sample_images(net_ema, n=64, device=device)
                vutils.save_image(imgs, os.path.join(run_dir, f"samples_epoch_{epoch}.png"), nrow=8, normalize=True, value_range=(-1, 1))

    print(f"Run artifacts saved to: {run_dir}")

# ------------------------------- Sampling ---------------------------------- #

@torch.no_grad()
def anisotropic_noise_like(x, s: int, sigma_R: float, sigma_P: float, sigma_iso: float) -> torch.Tensor:
    z = torch.randn_like(x)
    z2 = torch.randn_like(x)
    return sigma_R * R_apply(z, s) + sigma_P * P_apply(z, s) + sigma_iso * z2


@torch.no_grad()
def sample_images(net: nn.Module, n: int = 64, device: str = "cuda") -> torch.Tensor:
    net.eval()
    H = W = 32
    x = torch.randn(n, 3, H, W, device=device)

    # Multiscale schedule (coarse → fine)
    schedule: List[Dict] = [
        {"s": 16, "sigma_R": 0.5,  "sigma_P": 0.15, "sigma_iso": 0.02,  "a": 0.8, "steps": 150, "eta": 0.15},
        {"s": 8,  "sigma_R": 0.35, "sigma_P": 0.10, "sigma_iso": 0.02,  "a": 0.6, "steps": 150, "eta": 0.12},
        {"s": 4,  "sigma_R": 0.20, "sigma_P": 0.08,  "sigma_iso": 0.015, "a": 0.4, "steps": 120, "eta": 0.08},
        {"s": 1,  "sigma_R": 0.10, "sigma_P": 0.05,  "sigma_iso": 0.01,  "a": 0.2, "steps": 100, "eta": 0.05},
    ]

    for stage in schedule:
        s = stage["s"]
        sigma_R = stage["sigma_R"]
        sigma_P = stage["sigma_P"]
        sigma_iso = stage["sigma_iso"]
        a = stage["a"]
        steps = stage["steps"]
        eta = stage["eta"]

        # Build constant cond vector for this stage
        hp = {
            "a": torch.full((n,), a, device=device),
            "sigma_P": torch.full((n,), sigma_P, device=device),
            "sigma_R": torch.full((n,), sigma_R, device=device),
            "sigma_iso": torch.full((n,), sigma_iso, device=device),
            "s": torch.full((n,), float(s), device=device),
        }
        cond = build_cond_vector(hp)

        for _ in range(steps):
            score = net(x, cond)
            # Predictor: anisotropic Langevin step preconditioned by Σ
            step_update = Sigma_apply(score, s, sigma_R, sigma_P, sigma_iso)
            x = x + eta * step_update + math.sqrt(2.0 * eta) * anisotropic_noise_like(x, s, sigma_R, sigma_P, sigma_iso)
            x = x.clamp(-1, 1)

        # Optional projection nudging (reverse of forward bias)
        x = (1 + 0.05) * x - 0.05 * P_apply(x, s)

    return x

# ---------------------------------- Main ----------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root", type=str, default="./runs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", help="auto|mps|cuda|cpu")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = args.device
        # sanity checks
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[warn] MPS requested but not available; falling back to CPU")
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA requested but not available; falling back to CPU")
            device = "cpu"

    cfg = TrainConfig(
        data_root=args.data_root,
        out_root=args.out_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        samples_every=args.samples_every,
        save_every=args.save_every,
        seed=args.seed,
        device=device,
    )

    train(cfg)

if __name__ == "__main__":
    main()
