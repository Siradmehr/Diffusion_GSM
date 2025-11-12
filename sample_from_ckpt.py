# sample_from_ckpt.py
import torch
from torchvision.utils import save_image

# import your definitions from the training file
# (if your file is named differently, change the import)
from forward_hard_diffusion_pytorch import SmallUNet, sample_images

def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    net = SmallUNet(in_ch=6, base=64, emb_dim=256, cond_dim=8).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    # run the reverse process (coarse→fine)
    imgs = sample_images(net, n=args.num_samples, device=device)

    # save grid
    save_image(
        imgs,
        args.out,
        nrow=int(math.sqrt(args.num_samples)),
        normalize=True, value_range=(-1, 1)
    )
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
