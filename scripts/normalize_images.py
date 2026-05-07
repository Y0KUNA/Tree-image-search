#!/usr/bin/env python3
"""Normalize images in a directory tree.

Creates a mirror of the source directory under the destination path and
saves normalized images with the same filenames.

Behaviors:
- Resize images to target resolution (width x height) while preserving
  original aspect ratio, using one of three methods: pad, crop, or stretch.
- pad: scale to fit inside target and pad the remaining area with fill color.
- crop: scale to cover target and center-crop the excess.
- stretch: resize ignoring aspect ratio (may distort).

Example:
  python normalize_images.py --src Data --dst Data_clean --width 256 --height 256 --method pad --fill 0,0,0

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps


def parse_fill(s: str) -> Tuple[int, int, int]:
    parts = s.split(',')
    if len(parts) == 1:
        # allow hex like #RRGGBB
        p = parts[0].strip()
        if p.startswith('#') and len(p) == 7:
            r = int(p[1:3], 16)
            g = int(p[3:5], 16)
            b = int(p[5:7], 16)
            return (r, g, b)
        raise argparse.ArgumentTypeError('fill must be R,G,B or #RRGGBB')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('fill must be R,G,B')
    return tuple(int(p) for p in parts)


def normalize_image(img: Image.Image, target: Tuple[int, int], method: str, fill: Tuple[int, int, int]) -> Image.Image:
    target_w, target_h = target
    img = ImageOps.exif_transpose(img)
    # convert to RGB for consistent saving (preserve alpha when padding with white? we flatten)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    src_w, src_h = img.size
    if method == 'stretch':
        return img.resize((target_w, target_h), Image.LANCZOS)

    # compute scale preserving aspect ratio
    scale_w = target_w / src_w
    scale_h = target_h / src_h

    if method == 'pad':
        scale = min(scale_w, scale_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # create background and paste centered
        background = Image.new('RGB', (target_w, target_h), fill)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        background.paste(img_resized, (paste_x, paste_y))
        return background

    if method == 'crop':
        scale = max(scale_w, scale_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # center crop
        left = (new_w - target_w) // 2
        upper = (new_h - target_h) // 2
        right = left + target_w
        lower = upper + target_h
        return img_resized.crop((left, upper, right, lower))

    raise ValueError(f'unknown method: {method}')


def is_image_file(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}


def mirror_subdirs_and_process(src: Path, dst: Path, width: int, height: int, method: str, fill: Tuple[int, int, int]) -> None:
    if not src.exists():
        raise FileNotFoundError(f'source path does not exist: {src}')
    dst.mkdir(parents=True, exist_ok=True)

    # Walk source tree
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        # relative path from src
        rel = root_path.relative_to(src)
        target_dir = dst.joinpath(rel)
        target_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_file = root_path / fname
            dst_file = target_dir / fname
            if not is_image_file(src_file):
                # copy non-image as-is (optional) - skip for now
                continue
            try:
                with Image.open(src_file) as im:
                    out = normalize_image(im, (width, height), method, fill)
                    # preserve format via extension
                    ext = src_file.suffix.lower()
                    save_kwargs = {}
                    if ext in ('.jpg', '.jpeg'):
                        save_kwargs['quality'] = 95
                        out = out.convert('RGB')
                    # ensure parent exists
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    out.save(dst_file, **save_kwargs)
            except Exception as e:
                print(f'Failed processing {src_file}: {e}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize images to a fixed resolution while preserving aspect ratio.')
    parser.add_argument('--src', required=True, help='source directory (will be walked recursively)')
    parser.add_argument('--dst', required=True, help='destination directory (mirror structure will be created)')
    parser.add_argument('--width', type=int, required=True, help='target width in pixels')
    parser.add_argument('--height', type=int, required=True, help='target height in pixels')
    parser.add_argument('--method', choices=('pad', 'crop', 'stretch'), default='pad', help='how to handle aspect ratio differences')
    parser.add_argument('--fill', type=parse_fill, default='0,0,0', help='fill color for pad method, e.g. "0,0,0" or "#FFFFFF"')

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    mirror_subdirs_and_process(src, dst, args.width, args.height, args.method, args.fill)


if __name__ == '__main__':
    main()
