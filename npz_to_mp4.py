#!/usr/bin/env python3
"""Convert .npz genframe files to .mp4 videos.

Usage:
    python npz_to_mp4.py genframes_out/*.npz
    python npz_to_mp4.py genframes_out/*.npz --output_dir mp4s/ --fps 1.0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def convert(npz_path, output_dir, fps):
    data = np.load(npz_path)
    frames = data['frames']
    name = Path(npz_path).stem
    out_path = output_dir / f'{name}.mp4'

    H, W = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path, len(frames)


def main():
    parser = argparse.ArgumentParser(description='Convert .npz genframe files to .mp4 videos.')
    parser.add_argument('npz_files', type=str, nargs='+', help='.npz files to convert')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output directory (default: same directory as each .npz)')
    parser.add_argument('--fps', type=float, default=1.0, help='Output video framerate (default: 1.0)')
    args = parser.parse_args()

    for npz_path in args.npz_files:
        out_dir = Path(args.output_dir) if args.output_dir else Path(npz_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path, n = convert(npz_path, out_dir, args.fps)
        print(f'{Path(npz_path).name}: {n} frames -> {out_path}')


if __name__ == '__main__':
    main()
