#!/usr/bin/env python3
"""Convert a Timescales train/ directory to a TLGAN-compatible zip dataset.

Timescales stores training frames as <seconds_offset>.jpeg in a flat directory.
This script produces a zip archive with resized frames and a dataset.json file
containing normalized timestamps in the format expected by TLGAN's
ImageFolderDataset._load_raw_labels().

Usage:
    python tlgan/preproc/from_timescales.py $TIMESCALE_DATA/rane_52wks/train \
        --resolution 256 --output rane_256x256_365hz.zip
"""

import argparse
import json
import math
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Convert Timescales dataset to TLGAN zip format.')
    parser.add_argument('train_dir', type=str, help='Path to Timescales train/ directory with <seconds_offset>.jpeg files')
    parser.add_argument('--resolution', type=int, default=256, help='Target resolution (square)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Dataset name (suffix _<res>x<res>_<days>hz.zip added automatically)')
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    assert train_dir.is_dir(), f'{train_dir} is not a directory'

    jpegs = sorted(train_dir.glob('*.jpeg'), key=lambda p: int(p.stem))
    assert jpegs, f'No .jpeg files found in {train_dir}'

    offsets = [int(p.stem) for p in jpegs]
    min_offset = min(offsets)
    max_offset = max(offsets)
    span = max_offset - min_offset
    assert span > 0, f'Need at least two distinct offsets (got min={min_offset}, max={max_offset})'
    num_days = math.ceil(span / 86400)
    norm_factor = num_days * 86400

    print(f'Found {len(jpegs)} frames, offset range [{min_offset}, {max_offset}]s, span={span}s, num_days={num_days}')

    name = args.output or train_dir.parent.name
    out_path = Path(f'{name}_{args.resolution}x{args.resolution}_{num_days}hz.zip')

    labels = []
    res = args.resolution

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, (jpeg_path, offset) in enumerate(zip(jpegs, offsets)):
            t_norm = offset / norm_factor
            fname = f'{offset}.jpg'

            img = Image.open(jpeg_path)
            if img.size != (res, res):
                img = img.resize((res, res), Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format='JPEG', quality=95)
            zf.writestr(fname, buf.getvalue())

            labels.append([fname, [t_norm]])

            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{len(jpegs)}')

        meta = {'num_days': num_days, 'date_start': 0}
        dataset_json = json.dumps({'meta': meta, 'labels': labels})
        zf.writestr('dataset.json', dataset_json)

    print(f'Wrote {len(labels)} frames to {out_path}')
    print(f'num_days = {num_days}  (record this for the TLGAN prior config)')


if __name__ == '__main__':
    main()
