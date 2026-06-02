#!/usr/bin/env python3
"""Generate .npz genframes from a trained TLGAN for timescales evaluation.

Runs in the TLGAN conda environment. Produces per-clip .npz files
consumable by timescales evaluation via --genframes.

Usage:
    conda activate tlgan
    python generate_for_timescales.py \
        --pkl /path/to/network-snapshot-XXXXXX.pkl \
        --dataset_dir /cluster/.../TIMESCALE_DATA/rane_52wks \
        --split test_dispersed \
        --output_dir ./genframes_out
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

tlgan_root = str(Path(__file__).resolve().parent)
if tlgan_root not in sys.path:
    sys.path.insert(0, tlgan_root)


def discover_clips(dataset_dir, split):
    split_dir = Path(dataset_dir) / split
    assert split_dir.exists(), f'{split_dir} not found'
    clips = sorted(
        (d for d in split_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name),
    )
    assert clips, f'no clips in {split_dir}'
    return [d.name for d in clips]


def compute_sample_times(clip_name, fps, frames_per_clip, sample_interval):
    t_start = float(clip_name)
    duration = frames_per_clip / fps
    return np.arange(t_start, t_start + duration + sample_interval, sample_interval)


def generate_frames(G, sample_times, num_days, z_fixed, truncation_psi, noise_mode, batch_size):
    out_rect = getattr(G.synthesis, 'out_rect', None)
    synth_c_dim = G.synthesis.c_dim
    all_frames = np.zeros((len(sample_times), 256, 256, 3), dtype=np.uint8)

    with torch.no_grad():
        for i in range(0, len(sample_times), batch_size):
            batch_t = sample_times[i:i + batch_size]
            B = len(batch_t)

            t_norm = torch.tensor(
                batch_t / (num_days * 86400.0),
                dtype=torch.float32, device='cuda',
            ).unsqueeze(-1)

            z = z_fixed.expand(B, -1)
            cs = G.cond_xform(t_norm, broadcast=True)
            ws = G.mapping(z, cs[:, 0, :], truncation_psi=truncation_psi)
            img = G.synthesis(ws, cs[:, :, :synth_c_dim], noise_mode=noise_mode)

            if out_rect is not None:
                x1, y1, x2, y2 = out_rect
                img = img[:, :, y1:y2, x1:x2]
                if img.shape[-2] != 256 or img.shape[-1] != 256:
                    img = torch.nn.functional.interpolate(
                        img, size=(256, 256), mode='bilinear', align_corners=False,
                    )

            img = img.clamp(-1, 1)
            rgb = ((img.cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            for j in range(B):
                all_frames[i + j] = rgb[j].transpose(1, 2, 0)

    return all_frames


def main():
    parser = argparse.ArgumentParser(description='Generate .npz genframes from a trained TLGAN for timescales evaluation.')
    parser.add_argument('--pkl', type=str, required=True, help='Path to TLGAN network snapshot .pkl')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Timescales dataset root (e.g. TIMESCALE_DATA/rane_52wks). Only clip directory names are read to determine timestamps; no images are loaded.')
    parser.add_argument('--split', type=str, required=True, help='Split subdirectory to scan for clip names (test_dispersed, test_after, train_clips)')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory for .npz files (one per clip)')
    parser.add_argument('--sample_interval', type=float, default=1.0, help='Seconds between generated samples (default: 1.0)')
    parser.add_argument('--fps', type=float, default=30.0, help='Clip framerate (default: 30.0)')
    parser.add_argument('--frames_per_clip', type=int, default=1000, help='Frames per clip (default: 1000)')
    parser.add_argument('--seed', type=int, default=0, help='Fixed z seed (default: 0)')
    parser.add_argument('--truncation_psi', type=float, default=1.0, help='Mapping network truncation (default: 1.0)')
    parser.add_argument('--noise_mode', type=str, default='const', choices=['const', 'none'], help='Synthesis noise mode (default: const)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generation (default: 64)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_names = discover_clips(args.dataset_dir, args.split)

    print(f'Loading {args.pkl}')
    with open(args.pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].eval().requires_grad_(False).cuda()

    num_days = G.cond_args.num_days
    assert num_days and num_days > 0, f'num_days not found in pickle cond_args (got {num_days})'

    rng = np.random.RandomState(args.seed)
    z_fixed = torch.from_numpy(rng.standard_normal([1, G.z_dim]).astype(np.float32)).cuda()

    print(f'Generating genframes for {len(clip_names)} clips')
    print(f'  num_days={num_days}  sample_interval={args.sample_interval}s  seed={args.seed}')

    t0 = time.perf_counter()
    total_samples = 0

    for idx, clip_name in enumerate(clip_names):
        sample_times = compute_sample_times(clip_name, args.fps, args.frames_per_clip, args.sample_interval)
        frames = generate_frames(G, sample_times, num_days, z_fixed, args.truncation_psi, args.noise_mode, args.batch_size)

        out_path = output_dir / f'{int(clip_name):010d}.npz'
        np.savez_compressed(out_path, frames=frames, times=sample_times, sample_interval=args.sample_interval)

        total_samples += len(sample_times)
        print(f'  [{idx + 1}/{len(clip_names)}] {clip_name}: {len(sample_times)} samples -> {out_path.name}')

    elapsed = time.perf_counter() - t0
    print(f'\nDone: {total_samples} samples in {elapsed:.1f}s ({total_samples / elapsed:.1f} samples/s)')


if __name__ == '__main__':
    main()
