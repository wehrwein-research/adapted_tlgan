#!/usr/bin/env python3
"""Generate .mp4 videos from a trained TLGAN, one per clip.

Same clip discovery and frame generation as generate_for_timescales.py,
but writes .mp4 videos instead of .npz files.

Usage:
    conda activate tlgan
    python generate_mp4.py \
        --pkl /path/to/network-snapshot-XXXXXX.pkl \
        --dataset_dir /cluster/.../TIMESCALE_DATA/rane_52wks \
        --split test_dispersed \
        --output_dir ./mp4s_out
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

tlgan_root = str(Path(__file__).resolve().parent)
if tlgan_root not in sys.path:
    sys.path.insert(0, tlgan_root)

from generate_for_timescales import compute_sample_times, discover_clips, generate_frames


def write_mp4(frames, output_path, fps):
    H, W = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description='Generate .mp4 videos from a trained TLGAN, one per clip.')
    parser.add_argument('--pkl', type=str, required=True, help='Path to TLGAN network snapshot .pkl')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Timescales dataset root (e.g. TIMESCALE_DATA/rane_52wks). Only clip directory names are read to determine timestamps; no images are loaded.')
    parser.add_argument('--split', type=str, required=True, help='Split subdirectory to scan for clip names (test_dispersed, test_after, train_clips)')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory for .mp4 files (one per clip)')
    parser.add_argument('--video_fps', type=float, default=30.0, help='Output video framerate (default: 1.0)')
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

    print(f'Generating mp4s for {len(clip_names)} clips')
    print(f'  num_days={num_days}  sample_interval={args.sample_interval}s  seed={args.seed}')

    t0 = time.perf_counter()
    total_samples = 0

    for idx, clip_name in enumerate(clip_names):
        sample_times = compute_sample_times(clip_name, args.fps, args.frames_per_clip, args.sample_interval)
        frames = generate_frames(G, sample_times, num_days, z_fixed, args.truncation_psi, args.noise_mode, args.batch_size)

        out_path = output_dir / f'{int(clip_name):010d}.mp4'
        write_mp4(frames, out_path, args.video_fps)

        total_samples += len(sample_times)
        print(f'  [{idx + 1}/{len(clip_names)}] {clip_name}: {len(sample_times)} samples -> {out_path.name}')

    elapsed = time.perf_counter() - t0
    print(f'\nDone: {total_samples} samples in {elapsed:.1f}s ({total_samples / elapsed:.1f} samples/s)')


if __name__ == '__main__':
    main()
