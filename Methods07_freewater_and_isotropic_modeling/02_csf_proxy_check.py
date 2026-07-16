"""
HCP S1200 AUD twin study
==================================================
Goal
----
Test whether the between-family olfactory/reward diffusivity signal is a familial
partial-volume / CSF characteristic rather than axonal or exposure-driven.

Have per subject, the MSMT-CSD CSF compartment from phase3populationtemplate.py:
    processed/{Zyg}/{Pair}/sub-{sid}_{F|M}/csffod_norm.mif
Its 0th-order SH coefficient (volume 0) is the apparent CSF signal fraction — a
per-subject partial-volume / free-water proxy, in native space.

Output: BASE/csf_metrics/csf_partial_volume_metrics.csv  (merges on Subject)
"""

import os, glob, re, argparse, subprocess, logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

BASE          = os.environ.get("PROJECT_ROOT", ".")
PROCESSED_DIR = os.path.join(BASE, "processed")
REG_DIR       = os.path.join(BASE, "registration")
VOXEL_MASK_DIR= os.path.join(BASE, "final_masks", "voxel")
OUT_DIR       = os.path.join(BASE, "csf_metrics")
META_CSV      = os.path.join(BASE, "twintables", "network_roi_metrics_FINAL.csv")
OUT_CSV       = os.path.join(OUT_DIR, "csf_partial_volume_metrics.csv")
FS_SEARCH_ROOTS = [BASE]
NETWORKS      = ["Reward", "Salience", "DMN", "Olfactory"]
TEMPLATE_DIMS = (137, 176, 139)
MAX_WORKERS   = 12
WORKDIR       = "/dev/shm"

os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, f"phaseA_csf_{datetime.now():%Y%m%d_%H%M}.log")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger()


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def find_subject_dir(sid):
    hits = glob.glob(os.path.join(PROCESSED_DIR, "*", "*", f"sub-{sid}_*"))
    hits = [h for h in hits if os.path.isdir(h)]
    return hits[0] if hits else None


def brain_volume_mm3(mask_mif):
    ok, out, _ = run(["mrstats", mask_mif, "-output", "count", "-ignorezero"])
    if not ok:
        return np.nan
    try:
        nvox = float(out.strip().split()[0])
    except Exception:
        return np.nan
    ok, sp, _ = run(["mrinfo", mask_mif, "-spacing"])
    if not ok:
        return np.nan
    try:
        dx, dy, dz = [float(s) for s in sp.strip().split()[:3]]
    except Exception:
        return np.nan
    return nvox * dx * dy * dz


def extract_subject(sid):
    row = {"Subject": str(sid)}
    for net in NETWORKS:
        row[f"CSFfrac_{net}"] = np.nan
    row["CSFfrac_GlobalMean"] = np.nan
    row["BrainVol_mm3"]       = np.nan

    sdir = find_subject_dir(sid)
    if sdir is None:
        return row
    csffod = os.path.join(sdir, "csffod_norm.mif")
    mask   = os.path.join(sdir, "mask.mif")
    warp   = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
    if not (os.path.exists(csffod) and os.path.exists(warp)):
        return row

    ws = os.path.join(WORKDIR, f"csf_{sid}")
    os.makedirs(ws, exist_ok=True)
    try:
        csf0_native = os.path.join(ws, "csf0_native.mif")
        ok, _, e = run(["mrconvert", csffod, "-coord", "3", "0", "-axes", "0,1,2",
                        csf0_native, "-force"])
        if not ok:
            log.warning(f"{sid}: mrconvert csf l=0 failed: {e[:160]}")
            return row

        if os.path.exists(mask):
            ok, out, _ = run(["mrstats", csf0_native, "-mask", mask,
                              "-ignorezero", "-output", "mean"])
            if ok and out.strip():
                row["CSFfrac_GlobalMean"] = float(out.strip().split()[0])
            row["BrainVol_mm3"] = brain_volume_mm3(mask)

        csf0_tmpl = os.path.join(ws, "csf0_template.mif")
        ok, _, e = run(["mrtransform", csf0_native, "-warp", warp,
                        "-reorient_fod", "no", csf0_tmpl, "-force"])
        if not ok:
            log.warning(f"{sid}: warp failed: {e[:160]}")
            return row

        for net in NETWORKS:
            nmask = os.path.join(VOXEL_MASK_DIR, f"{net}_voxel_mask.mif")
            if not os.path.exists(nmask):
                continue
            ok, out, _ = run(["mrstats", csf0_tmpl, "-mask", nmask,
                              "-ignorezero", "-output", "mean"])
            if ok and out.strip():
                row[f"CSFfrac_{net}"] = float(out.strip().split()[0])
        return row
    finally:
        try:
            for f in glob.glob(os.path.join(ws, "*")):
                os.remove(f)
            os.rmdir(ws)
        except Exception:
            pass


def find_freesurfer():
    found = []
    for root in FS_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        found += glob.glob(os.path.join(root, "**", "stats", "aseg.stats"),
                           recursive=True)
    return sorted(set(found))


def parse_aseg(aseg_path):
    out = {}
    want_struct = {
        "Left-Lateral-Ventricle":  "FS_LLatVent",
        "Right-Lateral-Ventricle": "FS_RLatVent",
        "3rd-Ventricle":           "FS_3rdVent",
        "4th-Ventricle":           "FS_4thVent",
        "CSF":                     "FS_CSF",
    }
    try:
        with open(aseg_path) as fh:
            for line in fh:
                if line.startswith("# Measure EstimatedTotalIntraCranialVol"):
                    out["FS_eTIV"] = float(line.rstrip().split(",")[-2])
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[4] in want_struct:
                    out[want_struct[parts[4]]] = float(parts[3])
    except Exception:
        return None
    if "FS_LLatVent" in out and "FS_RLatVent" in out:
        out["FS_LatVent_total"] = out["FS_LLatVent"] + out["FS_RLatVent"]
    return out


def subject_from_aseg_path(p):
    m = re.search(r"(\d{6})", p)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subject", type=str, default=None)
    args = ap.parse_args()

    meta = pd.read_csv(META_CSV, low_memory=False)
    meta["Subject"] = meta["Subject"].astype(str)
    sids = [args.subject] if args.subject else list(meta["Subject"])

    if args.dry_run:
        n_ok = 0
        for sid in sids:
            sdir = find_subject_dir(sid)
            csf  = os.path.join(sdir, "csffod_norm.mif") if sdir else ""
            warp = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
            ok = sdir and os.path.exists(csf) and os.path.exists(warp)
            n_ok += bool(ok)
        fs = find_freesurfer()
        log.info(f"Dry-run: {n_ok}/{len(sids)} subjects have CSF + warp inputs")
        return

    log.info(f"Extracting CSF partial-volume proxies for {len(sids)} subjects")
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(extract_subject, sid): sid for sid in sids}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result()); done += 1
    csf_df = pd.DataFrame(rows)

    fs = find_freesurfer()
    if fs:
        fsrows = []
        for p in fs:
            sid = subject_from_aseg_path(p)
            if not sid:
                continue
            vals = parse_aseg(p)
            if vals:
                vals["Subject"] = sid
                fsrows.append(vals)
        if fsrows:
            fs_df = pd.DataFrame(fsrows).drop_duplicates("Subject")
            csf_df = csf_df.merge(fs_df, on="Subject", how="left")

    out = meta.merge(csf_df, on="Subject", how="left")
    out.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_CSV}  ({len(out)} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
