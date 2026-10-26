#!/usr/bin/env python3
"""Reproduce the WS-DINO BBBC021 preprocessing in Python (no CellProfiler).

Mirrors WS-DINO/preprocessing/*.cppipe: per-plate, per-channel illumination
function = mean over all images on the plate, median-smoothed (CP filter 320 px
at 2x resampling), rescaled to >=1; each image divided by it, resized by 0.5
(1280x1024 -> 640x512), saved as 16-bit TIFF named *_corrected_resized.tiff.
Then rewrites the WS-DINO annotated/DMSO CSVs to point at the new files.

Usage: python preprocess_bbbc021.py --raw-dir RAW --wsdino-repo WSDINO --out-dir OUT [--workers N]
RAW holds the unzipped plate folders (e.g. RAW/Week1_22123/*.tif).
"""
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, tifffile, cv2
from scipy.ndimage import median_filter

CH_PREFIX = {"DAPI": "w1", "Tubulin": "w2", "Actin": "w4"}


def illum_function(files):
    acc = None
    for f in files:
        im = tifffile.imread(f).astype(np.float64)
        acc = im if acc is None else acc + im
    mean = acc / len(files)
    # CP: resample by 2 then median filter size 320 (in full-res px). We go
    # coarser (factor 8, size 40) for speed; the result is a smooth field.
    h, w = mean.shape
    small = cv2.resize(mean, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    small = median_filter(small, size=40, mode="nearest")
    illum = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # CP "rescale illumination function": divide by robust minimum, floor at 1
    rmin = np.percentile(illum[illum > 0], 2) if (illum > 0).any() else 1.0
    illum = np.maximum(illum / rmin, 1.0)
    return illum


def process_plate(args):
    plate_dir, out_root, needed = args
    plate_dir = Path(plate_dir)
    out_dir = Path(out_root) / plate_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = sorted(plate_dir.glob("*.tif"))
    n = 0
    for ch, pre in CH_PREFIX.items():
        files = [f for f in tifs if f"_{pre}" in f.name]
        if not files:
            continue
        illum = illum_function(files)
        for f in files:
            if needed is not None and f.stem not in needed:
                continue
            outp = out_dir / f"{f.stem}_corrected_resized.tiff"
            if outp.exists():
                n += 1
                continue
            im = tifffile.imread(f).astype(np.float64) / illum
            h, w = im.shape
            im = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            tifffile.imwrite(outp, np.clip(im, 0, 65535).astype(np.uint16))
            n += 1
    return plate_dir.name, n


def rewrite_csv(src, dst, out_root):
    df = pd.read_csv(src)
    for ch in ["DAPI", "Tubulin", "Actin"]:
        col = f"Image_FileName_{ch}"
        df[col] = [str(Path(out_root) / Path(p).parent.name / Path(p).name) for p in df[col]]
    df.to_csv(dst, index=False)
    missing = sum(not os.path.exists(p) for ch in ["DAPI", "Tubulin", "Actin"]
                  for p in df[f"Image_FileName_{ch}"])
    print(f"{dst}: {len(df)} rows, missing files: {missing}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--wsdino-repo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out_dir).resolve(); img_root = out / "images"; img_root.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(Path(a.wsdino_repo) / "BBBC021_annotated.csv")
    dmso = pd.read_csv(Path(a.wsdino_repo) / "BBBC021_DMSO.csv")
    needed = {}
    for df in (ann, dmso):
        for ch in ["DAPI", "Tubulin", "Actin"]:
            for p in df[f"Image_FileName_{ch}"]:
                p = Path(p); plate = p.parent.name
                needed.setdefault(plate, set()).add(p.name.replace("_corrected_resized.tiff", ""))
    plates = [d for d in sorted(Path(a.raw_dir).iterdir()) if d.is_dir() and d.name in needed]
    print(f"{len(plates)} plates to process (need {len(needed)})", flush=True)
    jobs = [(str(d), str(img_root), needed[d.name]) for d in plates]
    with ProcessPoolExecutor(a.workers) as ex:
        for name, n in ex.map(process_plate, jobs):
            print(f"  {name}: {n} images", flush=True)
    m1 = rewrite_csv(Path(a.wsdino_repo) / "BBBC021_annotated.csv", out / "BBBC021_annotated_wsdino.csv", img_root)
    m2 = rewrite_csv(Path(a.wsdino_repo) / "BBBC021_DMSO.csv", out / "BBBC021_DMSO_wsdino.csv", img_root)
    sys.exit(1 if (m1 or m2) else 0)


if __name__ == "__main__":
    main()
