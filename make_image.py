import pickle
import argparse
import numpy as np
import torch
from PIL import Image


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate a single image from a time-conditioned GAN pickle."
    )
    parser.add_argument(
        "network_pickle",
        type=str,
        help="Path to the trained GAN .pkl file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for latent sampling (controls identity). Default: 0.",
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=1.0,
        help="Truncation psi for the mapping network (0–1). Default: 1.0.",
    )
    parser.add_argument(
        "--t-trend",
        type=float,
        default=0.5,
        help="Global trend position in [0, 1]. 0 = start of dataset, 1 = end. Default: 0.5.",
    )
    parser.add_argument(
        "--t-year",
        type=float,
        default=0.5,
        help="Day of year in [0, 1]. Tip: pass (day_number / 365) e.g. 0.548 ≈ day 200. Default: 0.5.",
    )
    parser.add_argument(
        "--t-day",
        type=float,
        default=0.5,
        help="Time of day in [0, 1]. Tip: pass (hour / 24) e.g. 0.5 = noon. Default: 0.5.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save the generated image. Default: output.png.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on. Default: cuda.",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="const",
        choices=["const", "random", "none"],
        help="Noise mode passed to the synthesis network. Default: const.",
    )

    args = parser.parse_args()

    return {
        "pkl":        args.network_pickle,
        "seed":       args.seed,
        "truncation": args.truncation,
        "t_trend":    args.t_trend,
        "t_year":     args.t_year,
        "t_day":      args.t_day,
        "output":     args.output,
        "device":     args.device,
        "noise_mode": args.noise_mode,
    }


def load_model(cfg: dict):
    with open(cfg["pkl"], "rb") as f:
        data = pickle.load(f)
    G = data["G_ema"].to(cfg["device"]).eval()
    return G


def build_conditioning(G, cfg: dict) -> torch.Tensor:
    device = cfg["device"]
    has_fourier = (
        hasattr(G, "cond_xform")
        and G.init_kwargs.get("cond_args", {}).get("type") in ["fourier", "f_concat"]
    )

    if has_fourier:
        freqs = G.cond_xform.get_frequencies()
        n_freq = len(freqs)

        t_values = [cfg["t_trend"], cfg["t_year"], cfg["t_day"]]
        if n_freq == 2:
            t_values = [cfg["t_trend"], cfg["t_day"]]
        elif n_freq != 3:
            raise ValueError(
                f"Unexpected number of Fourier frequencies: {n_freq}. "
                "Expected 2 (trend+day) or 3 (trend+year+day)."
            )

        t_raw = torch.tensor([t_values], dtype=torch.float32, device=device)
        cs = t_raw.unsqueeze(1).repeat_interleave(G.num_ws, dim=1)
        cs = G.cond_xform(cs, broadcast=False)

    else:
        t_linear = cfg["t_trend"]
        cs = t_linear * torch.ones([1, G.c_dim], dtype=torch.float32, device=device)
        cs = cs.unsqueeze(1).repeat_interleave(G.num_ws, dim=1)

    return cs


def generate_image(G, cs: torch.Tensor, cfg: dict) -> np.ndarray:
    device = cfg["device"]

    rng = np.random.RandomState(cfg["seed"])
    z = torch.from_numpy(
        rng.standard_normal([1, G.z_dim]).astype(np.float32)
    ).to(device)

    ws = G.mapping(
        z,
        cs[:, 0, :],
        truncation_psi=cfg["truncation"],
        truncation_cutoff=G.num_ws,
    )

    synth_c_dim = G.synthesis.c_dim
    for i in range(10):
        print(f"Making image {i}")
        img_tensor = G.synthesis(
            ws,
            cs[:, :, 0:synth_c_dim],
            noise_mode=cfg["noise_mode"],
        )

    if hasattr(G.synthesis, "out_rect"):
        x1, y1, x2, y2 = G.synthesis.out_rect
        img_tensor = img_tensor[:, :, y1:y2, x1:x2]

    img_np = (
        img_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5
    ).clip(0, 255).astype(np.uint8)

    return img_np


def main() -> None:
    cfg = parse_args()

    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    print(f"Loading model from: {cfg['pkl']}")
    G = load_model(cfg)

    print(f"Conditioning → trend={cfg['t_trend']:.4f}  year={cfg['t_year']:.4f}  day={cfg['t_day']:.4f}")
    cs = build_conditioning(G, cfg)

    print(f"Generating image (seed={cfg['seed']}, truncation={cfg['truncation']})…")
    img_np = generate_image(G, cs, cfg)

    Image.fromarray(img_np).save(cfg["output"])
    print(f"Saved → {cfg['output']}")


if __name__ == "__main__":
    main()
