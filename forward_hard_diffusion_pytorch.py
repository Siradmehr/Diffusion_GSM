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
from dataclasses import dataclass
from typing import Tuple, Dict, List

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
        # x: (B, features) real-valued conditioning (we'll project to [0,1] range)
        # Produce a sinusoidal set per feature and concatenate
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
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, emb_dim=128, cond_dim=8):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = ResBlock(base, base, emb_dim)
        self.down2 = ResBlock(base, base * 2, emb_dim)
        self.pool1 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)

        self.mid1 = ResBlock(base * 2, base * 2, emb_dim)
        self.mid2 = ResBlock(base * 2, base * 2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)
        self.upb1 = ResBlock(base * 2, base, emb_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

        # sinusoidal embedding for cond
        self.sin_emb = SinusoidalEmbedding(dim=cond_dim * 16)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 16, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, cond_vec):
        # cond_vec: (B, cond_dim) — we'll sinusoidally embed then MLP
        emb = self.sin_emb(cond_vec)
        emb = self.cond_mlp(emb)

        h = self.in_conv(x)
        h = self.down1(h, emb)
        h = self.down2(h, emb)
        h = self.pool1(h)
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)
        h = self.up1(h)
        h = self.upb1(h, emb)
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 32
    save_every: int = 1
    samples_every: int = 1
    seed: int = 42


# Multiscale grid sizes for CIFAR-10
S_LIST = [1, 4, 8, 16]  # 1 keeps identity; others impose block means


def sample_training_hyperparams(batch: int, device: str) -> Dict[str, torch.Tensor]:
    """Randomize (per batch) the corruption parameters.
    - a in [0.2, 0.8]
    - sigma_R > sigma_P, both in [0.05, 0.5]
    - sigma_iso small in [0.0, 0.05]
    - s from S_LIST (scalar, but we pass normalized embedding)
    """
    a = torch.rand(batch, device=device) * 0.6 + 0.2
    sigma_P = torch.rand(batch, device=device) * 0.25 + 0.05
    sigma_R = sigma_P + torch.rand(batch, device=device) * 0.25 + 0.05
    sigma_iso = torch.rand(batch, device=device) * 0.05
    s_idx = torch.randint(low=0, high=len(S_LIST), size=(batch,), device=device)
    s_vals = torch.tensor(S_LIST, device=device, dtype=torch.float32)[s_idx]

    return {
        "a": a,
        "sigma_P": sigma_P,
        "sigma_R": sigma_R,
        "sigma_iso": sigma_iso,
        "s": s_vals,
    }


def build_cond_vector(hp: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Build conditioning vector per sample: [a, sigma_P, sigma_R, sigma_iso, log2(s)/4, 1/s, sigma_R/sigma_P, a*sigma_R]
    Keep values in roughly [0,1] ranges.
    """
    a = hp["a"].unsqueeze(1)
    sigma_P = hp["sigma_P"].unsqueeze(1)
    sigma_R = hp["sigma_R"].unsqueeze(1)
    sigma_iso = hp["sigma_iso"].unsqueeze(1)
    s = hp["s"].unsqueeze(1)
    cond = torch.cat([
        a,
        sigma_P,
        sigma_R,
        sigma_iso,
        torch.log2(s) / 5.0,  # log-scale of block size
        1.0 / s,
        sigma_R / (sigma_P + 1e-6),
        a * sigma_R,
    ], dim=1)
    return cond

# ------------------------------- Training ---------------------------------- #

def make_dataloader(cfg: TrainConfig) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1,1]
    ])
    ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_root, exist_ok=True)
    run_dir = os.path.join(cfg.out_root, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    device = cfg.device
    loader = make_dataloader(cfg)
    net = SmallUNet(in_ch=3, base=64, emb_dim=192, cond_dim=8).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        ema_loss = None
        for (x0, _) in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            hp = sample_training_hyperparams(B, device)
            cond = build_cond_vector(hp)

            # Forward one step: x1 = m + noise
            x1, m = forward_one_step(
                x0, s=int(hp["s"][0].item()), a=float(hp["a"][0].item()),
                sigma_R=float(hp["sigma_R"][0].item()), sigma_P=float(hp["sigma_P"][0].item()), sigma_iso=float(hp["sigma_iso"][0].item()),
            )
            # NOTE: for simplicity we sample one shared (s, a, sigmas) for the whole batch.

            # Target score: -Sigma^{-1}(x1 - m)
            target = -Sigma_inv_apply(
                x1 - m,
                s=int(hp["s"][0].item()),
                sigma_R=float(hp["sigma_R"][0].item()),
                sigma_P=float(hp["sigma_P"][0].item()),
                sigma_iso=float(hp["sigma_iso"][0].item()),
            )

            pred = net(x1, cond)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            global_step += 1
            ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
            pbar.set_postfix(loss=f"{ema_loss:.4f}")

        # Save checkpoint
        torch.save({
            "model": net.state_dict(),
            "cfg": cfg.__dict__,
        }, os.path.join(run_dir, f"ckpt_epoch_{epoch}.pt"))

        if epoch % cfg.samples_every == 0:
            with torch.no_grad():
                net.eval()
                imgs = sample_images(net, n=64, device=device)
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
        {"s": 16, "sigma_R": 0.5, "sigma_P": 0.15, "sigma_iso": 0.02, "a": 0.8, "steps": 60, "eta": 0.8},
        {"s": 8,  "sigma_R": 0.35, "sigma_P": 0.10, "sigma_iso": 0.02, "a": 0.6, "steps": 60, "eta": 0.6},
        {"s": 4,  "sigma_R": 0.20, "sigma_P": 0.08,  "sigma_iso": 0.015, "a": 0.4, "steps": 60, "eta": 0.4},
        {"s": 1,  "sigma_R": 0.10, "sigma_P": 0.05,  "sigma_iso": 0.01,  "a": 0.2, "steps": 60, "eta": 0.3},
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root", type=str, default="./runs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    )

    train(cfg)

if __name__ == "__main__":
    main()
