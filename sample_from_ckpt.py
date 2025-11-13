# sample_from_ckpt.py
import torch
from torchvision.utils import save_image
import math

# import your definitions from the training file
# (if your file is named differently, change the import)
from forward_hard_diffusion_pytorch import SmallUNet
import torch.nn.functional as F
def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def Sigma_apply(v, s, sigma_R, sigma_P, sigma_iso):
    Pv = P_apply(v, s); Rv = v - Pv
    return Rv * ((sigma_R**2)+(sigma_iso**2)) + Pv * ((sigma_P**2)+(sigma_iso**2))

def sample_images(net, n, device, in_ch, mode="ode"):
    net.eval()
    H=W=32; x = torch.randn(n,3,H,W,device=device)
    schedule = [
        {"s":16,"sigma_R":0.45,"sigma_P":0.12,"sigma_iso":0.02,"a":0.75,"steps":180,"eta":0.10},
        {"s": 8,"sigma_R":0.35,"sigma_P":0.10,"sigma_iso":0.02,"a":0.60,"steps":180,"eta":0.08},
        {"s": 4,"sigma_R":0.22,"sigma_P":0.08,"sigma_iso":0.015,"a":0.45,"steps":150,"eta":0.06},
        {"s": 1,"sigma_R":0.12,"sigma_P":0.05,"sigma_iso":0.010,"a":0.30,"steps":150,"eta":0.05},
    ]
    for stage in schedule:
        s = stage["s"]; sR=stage["sigma_R"]; sP=stage["sigma_P"]; sI=stage["sigma_iso"]; a=stage["a"]; steps=stage["steps"]; eta0=stage["eta"]
        cond = build_cond(a, sP, sR, sI, s, n, device)
        for t in range(steps):
            eta = eta0 * (0.999 ** t)      # small decay within stage
            x_in = torch.cat([x, P_apply(x,s)],1) if in_ch==6 else x
            score = net(x_in, cond)
            step = Sigma_apply(score, s, sR, sP, sI)
            if mode == "ode":
                x = x + eta * step          # deterministic probability-flow style
            else:
                x = x + eta * step + math.sqrt(max(1e-6,2.0*eta)) * anisotropic_noise_like(x, s, sR, sP, sI)
            # IMPORTANT: nudge TOWARD block means (stabilize coarse structure)
            eps = 0.05
            x = (1 - eps)*x + eps*P_apply(x, s)
            x = x.clamp(-1,1)
    return x
def main():
    import argparse, math
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to ckpt .pt (use the *_ema.pt for best samples)",default="/Users/apple/Diffusion_GSM/ckpt_epoch_35.pt")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--out", default="samples_from_ckpt.png")
    args = p.parse_args()

    device = pick_device()
    print(f"[info] using device: {device}")

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # NOTE: match the model args you trained with (in_ch=6 if you used the extra block-mean channel)
    net = SmallUNet(in_ch=3, base=64, emb_dim=192, cond_dim=8).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    # run the reverse process (coarse→fine)
    imgs = sample_images(net, n=args.num_samples, device=device,in_ch=3, mode="ode")

    # save grid
    save_image(
        imgs,
        args.out,
        nrow=int(math.sqrt(args.num_samples)),
        normalize=True, value_range=(-1, 1)
    )
    print(f"[done] wrote {args.out}")


# sample_from_ckpt_auto.py
def build_cond(a, sP, sR, sIso, s, n, device):
    s_t = torch.full((n,1), float(s), device=device)
    return torch.cat([
        torch.full((n,1), float(a),     device=device),
        torch.full((n,1), float(sP),    device=device),
        torch.full((n,1), float(sR),    device=device),
        torch.full((n,1), float(sIso),  device=device),
        torch.log2(s_t)/5.0,
        1.0/s_t,
        torch.full((n,1), float(sR/sP), device=device),
        torch.full((n,1), float(a*sR),  device=device),
    ],1)

@torch.no_grad()
def anisotropic_noise_like(x, s, sigma_R, sigma_P, sigma_iso):
    z = torch.randn_like(x); z2 = torch.randn_like(x)
    return sigma_R*R_apply(z, s) + sigma_P*P_apply(z, s) + sigma_iso*z2


# --- projections ---
def block_project(x, s):
    if s == 1: return x
    y = F.avg_pool2d(x, kernel_size=s, stride=s)
    return F.interpolate(y, scale_factor=s, mode="nearest")
def P_apply(x, s): return block_project(x, s)
def R_apply(x, s): return x - block_project(x, s)

def Sigma_apply(v, s, sigma_R, sigma_P, sigma_iso):
    Pv = P_apply(v, s); Rv = v - Pv
    return Rv * ((sigma_R**2)+(sigma_iso**2)) + Pv * ((sigma_P**2)+(sigma_iso**2))

if __name__ == "__main__":
    main()
