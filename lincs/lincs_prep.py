#!/usr/bin/env python3
"""Build a BBBC021-style MoA benchmark from the LINCS Cell Painting collection (cpg0004, A549 cells).

Selection: 12 MoAs with strong expected phenotypes, up to 6 compounds per MoA with a single
annotated MoA, the two highest doses of each compound, replicate plates from imaging batches 1-3,
fields 1-4 of each well, and channels DNA, ER and AGP (mapped to the DAPI, Tubulin and Actin slots
of the pipeline). Images are the 8-bit compressed PNGs of the Cell Painting Gallery. Each image is
divided by a per-plate, per-channel illumination function (mean image of the downloaded fields,
median smoothed, rescaled to a minimum of 1), resized to 576 x 576 pixels, multiplied by 30 and
saved as 16-bit TIFF. The 'Batch' column holds the plate layout, because every compound was imaged
in every batch; NSCB therefore excludes neighbors that share the query's plate layout.

Usage: lincs_prep.py --meta META_DIR --out OUT_DIR [--workers N]
"""
import argparse, os, subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, tifffile, cv2
from PIL import Image
from scipy.ndimage import median_filter

MOAS = ["glucocorticoid receptor agonist", "topoisomerase inhibitor", "EGFR inhibitor", "HDAC inhibitor",
        "MEK inhibitor", "tubulin polymerization inhibitor", "PI3K inhibitor", "CDK inhibitor",
        "mTOR inhibitor", "HSP inhibitor", "RNA polymerase inhibitor", "DNA inhibitor"]
CH = {"DAPI": 1, "Tubulin": 2, "Actin": 4}      # DNA, ER, AGP
FIELDS = [1, 2, 3, 4]
BASE = "https://cellpainting-gallery.s3.amazonaws.com/cpg0004-lincs/broad/images/2016_04_01_a549_48hr_batch1_compressed/images"
SCALE = 30.0
SIZE = 576


def well_rc(w):
    return int(ord(w[0]) - 64), int(w[1:])


def fname(w, f, c):
    r, col = well_rc(w)
    return f"r{r:02d}c{col:02d}f{f:02d}p01-ch{c}sk1fk1fl1.png"


def select(meta):
    em = pd.read_csv(meta / "external_metadata.tsv", sep="\t")
    bp = pd.read_csv(meta / "barcode_platemap.csv")
    pm = pd.concat([pd.read_csv(f, sep="\t") for f in sorted((meta / "platemap").glob("*.txt"))])
    em = em[em.moa.isin(MOAS)]
    trt = pm[pm.broad_sample.isin(em.broad_sample)].merge(em[["broad_sample", "pert_iname", "moa"]], on="broad_sample")
    trt["dose"] = trt.mmoles_per_liter.round(4)
    top = trt.groupby("pert_iname").dose.apply(lambda s: sorted(s.unique())[-2:]).to_dict()
    trt = trt[[d in top[n] for n, d in zip(trt.pert_iname, trt.dose)]]
    ok = trt.groupby("pert_iname").dose.nunique()
    trt = trt[trt.pert_iname.isin(ok[ok == 2].index)]
    keep = []
    for m, g in trt.groupby("moa"):
        keep += sorted(g.pert_iname.unique())[:6]
    trt = trt[trt.pert_iname.isin(keep)]
    bp = bp[bp.Batch_Number.isin([1, 2, 3])]
    trt = trt.merge(bp, left_on="plate_map_name", right_on="Plate_Map_Name")
    dm = pm[(pm.broad_sample.isna()) & (pm.solvent == "DMSO")]
    dm = dm[dm.plate_map_name.isin(trt.plate_map_name.unique())]
    dm = dm.groupby("plate_map_name").head(6).merge(bp, left_on="plate_map_name", right_on="Plate_Map_Name")
    return trt, dm


def plate_job(args):
    plate, rows, out = args
    raw = Path(out) / "raw" / plate; raw.mkdir(parents=True, exist_ok=True)
    img = Path(out) / "images" / plate; img.mkdir(parents=True, exist_ok=True)
    names = sorted({fname(w, f, c) for w in rows for f in FIELDS for c in CH.values()})
    need = [n for n in names if not (raw / n).exists()]
    if need:
        lst = raw / "urls.txt"
        lst.write_text("".join(f"{BASE}/{plate}/{n}\n" for n in need))
        subprocess.run(f"cd {raw} && xargs -P 16 -n 1 curl -s -f -O < urls.txt", shell=True)
    n = 0
    for ch, c in CH.items():
        files = [raw / fname(w, f, c) for w in rows for f in FIELDS]
        files = [p for p in files if p.exists()]
        if not files:
            continue
        acc = None
        for p in files:
            a = np.asarray(Image.open(p), dtype=np.float64)
            acc = a if acc is None else acc + a
        mean = acc / len(files)
        h, w = mean.shape
        small = cv2.resize(mean, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        small = median_filter(small, size=40, mode="nearest")
        illum = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        illum = illum / max(illum.min(), 1e-6)
        for p in files:
            dst = img / p.name.replace(".png", "_corrected_resized.tiff")
            if not dst.exists():
                a = np.asarray(Image.open(p), dtype=np.float64) / illum
                a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA) * SCALE
                tifffile.imwrite(dst, np.clip(a, 0, 65535).astype(np.uint16))
            n += 1
    return plate, n


def table(df, out, kind):
    out = Path(out)
    recs = []
    for _, r in df.iterrows():
        for f in FIELDS:
            rec = {f"Image_FileName_{ch}": str(out / "images" / r.Assay_Plate_Barcode /
                                                fname(r.well_position, f, c).replace(".png", "_corrected_resized.tiff"))
                   for ch, c in CH.items()}
            rec.update(Plate_Name=r.Assay_Plate_Barcode, Well=r.well_position, Field=f, Replicate=int(r.Batch_Number),
                       Layout=r.plate_map_name)
            if kind == "trt":
                rec.update(Image_Metadata_Compound=r.pert_iname, Image_Metadata_Concentration=r.dose, MoA=r.moa)
            recs.append(rec)
    t = pd.DataFrame(recs)
    t = t[[all(os.path.exists(t.loc[i, f"Image_FileName_{ch}"]) for ch in CH) for i in t.index]].reset_index(drop=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    trt, dm = select(Path(a.meta))
    print("MoAs", trt.moa.nunique(), "compounds", trt.pert_iname.nunique(), "treatments",
          trt.groupby(["pert_iname", "dose"]).ngroups, "plates", trt.Assay_Plate_Barcode.nunique(), flush=True)
    wells = pd.concat([trt[["Assay_Plate_Barcode", "well_position"]], dm[["Assay_Plate_Barcode", "well_position"]]])
    jobs = [(p, sorted(g.well_position.unique()), a.out) for p, g in wells.groupby("Assay_Plate_Barcode")]
    with ProcessPoolExecutor(a.workers) as ex:
        for p, n in ex.map(plate_job, jobs):
            print(" ", p, n, flush=True)
    T = table(trt, a.out, "trt"); D = table(dm, a.out, "dmso")
    codes = lambda s: pd.Series(pd.factorize(s, sort=True)[0], index=s.index)
    T["Unique_Compounds"] = codes(T.Image_Metadata_Compound)
    T["Unique_MoA"] = codes(T.MoA)
    T["Unique_Treatments"] = codes(T.Image_Metadata_Compound + "@" + T.Image_Metadata_Concentration.astype(str))
    allp = pd.concat([T.Plate_Name, D.Plate_Name])
    pc = {p: i for i, p in enumerate(sorted(allp.unique()))}
    lc = {l: i for i, l in enumerate(sorted(pd.concat([T.Layout, D.Layout]).unique()))}
    for t in (T, D):
        t["Plate"] = t.Plate_Name.map(pc); t["Batch"] = t.Layout.map(lc)
    T.to_csv(Path(a.out) / "BBBC021_annotated_wsdino.csv", index=False)
    D.to_csv(Path(a.out) / "BBBC021_DMSO_wsdino.csv", index=False)
    print("annotated", len(T), "dmso", len(D), "treatments", T.Unique_Treatments.nunique(),
          "compounds", T.Unique_Compounds.nunique(), "moas", T.Unique_MoA.nunique(), "layouts", T.Batch.nunique())


if __name__ == "__main__":
    main()
