"""
HCP S1200 AUD twin study
====================================================
Goal
----
Re-run the diffusivity analysis on FREE-WATER-ELIMINATED tensors. If the
between-family olfactory/reward diffusivity signal lives in the free-water
compartment, free-water correction removes it from FW-MD and it reappears in the
free-water fraction f.

HCP DWI is multi-shell (b = 1000/2000/3000), the regime where free-water DTI
is well posed. Per subject (native space, fit restricted to the four networks):
  1. export dwi_biascorr.mif -> NIfTI + FSL bvecs/bvals
  2. warp union of network masks (template -> native), dilate, intersect brain mask
  3. DIPY FreeWaterTensorModel.fit within fit_mask
  4. save FW-MD, FW-FA, FW-RD, FW-AD and free-water fraction f (native)
  5. warp each map native -> template
  6. mean within each network voxel mask

Output: BASE/final_stats/MASTER_FREEWATER_DTI.csv
"""

import os, glob, argparse, subprocess, logging, traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

BASE          = os.environ.get("PROJECT_ROOT", ".")
PROCESSED_DIR = os.path.join(BASE, "processed")
REG_DIR       = os.path.join(BASE, "registration")
VOXEL_MASK_DIR= os.path.join(BASE, "final_masks", "voxel")
FW_DIR        = os.path.join(BASE, "freewater")
META_CSV      = os.path.join(BASE, "twintables", "network_roi_metrics_FINAL.csv")
OUT_CSV       = os.path.join(BASE, "final_stats", "MASTER_FREEWATER_DTI.csv")

NETWORKS      = ["Reward", "Salience", "DMN", "Olfactory"]
N_WORKERS     = 12
FIT_METHOD    = "WLS"
DILATE_VOX    = 3
WORKDIR       = "/dev/shm"
FW_METRICS    = ["MD", "FA", "RD", "AD", "f"]

os.makedirs(FW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
LOG_FILE = os.path.join(FW_DIR, f"phaseB_fw_{datetime.now():%Y%m%d_%H%M}.log")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger()


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def find_subject_dir(sid):
    hits = [h for h in glob.glob(os.path.join(PROCESSED_DIR, "*", "*", f"sub-{sid}_*"))
            if os.path.isdir(h)]
    return hits[0] if hits else None


def build_template_union(ws):
    union = os.path.join(ws, "net_union_template.mif")
    masks = [os.path.join(VOXEL_MASK_DIR, f"{n}_voxel_mask.mif") for n in NETWORKS]
    masks = [m for m in masks if os.path.exists(m)]
    if not masks:
        return None
    tmp = os.path.join(ws, "net_sum.mif")
    ok, _, e = run(["mrmath"] + masks + ["max", tmp, "-force"])
    if not ok:
        log.error(f"union mrmath failed: {e[:160]}"); return None
    ok, _, e = run(["mrcalc", tmp, "0", "-gt", union, "-datatype", "bit", "-force"])
    if not ok:
        log.error(f"union binarise failed: {e[:160]}"); return None
    return union


def process_subject(sid, union_template):
    res = {"Subject": str(sid), "status": "?"}
    sdir = find_subject_dir(sid)
    if sdir is None:
        res["status"] = "no_dir"; return res
    dwi   = os.path.join(sdir, "dwi_biascorr.mif")
    nmask = os.path.join(sdir, "mask.mif")
    fwd   = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
    inv   = os.path.join(REG_DIR, f"sub-{sid}_warp_inv.mif")
    for p in (dwi, nmask, fwd, inv):
        if not os.path.exists(p):
            res["status"] = f"missing:{os.path.basename(p)}"; return res

    tmpl_out = {m: os.path.join(FW_DIR, "template", f"sub-{sid}_FW{m}_{FIT_METHOD}_tmpl.mif")
                for m in FW_METRICS}
    if all(os.path.exists(p) for p in tmpl_out.values()):
        res["status"] = "exists"; return _extract(sid, tmpl_out, res)

    ws = os.path.join(WORKDIR, f"fw_{sid}")
    os.makedirs(ws, exist_ok=True)
    os.makedirs(os.path.join(FW_DIR, "template"), exist_ok=True)
    try:
        from dipy.io.image import load_nifti, save_nifti
        from dipy.io.gradients import read_bvals_bvecs
        from dipy.core.gradients import gradient_table
        import dipy.reconst.fwdti as fwdti

        dwi_nii = os.path.join(ws, "dwi.nii.gz")
        fbval   = os.path.join(ws, "bvals")
        fbvec   = os.path.join(ws, "bvecs")
        ok, _, e = run(["mrconvert", dwi, dwi_nii, "-export_grad_fsl", fbvec, fbval, "-force"])
        if not ok:
            res["status"] = "mrconvert_fail"; return res

        union_nat = os.path.join(ws, "union_native.mif")
        ok, _, e = run(["mrtransform", union_template, "-warp", inv, "-interp", "nearest",
                        "-template", nmask, union_nat, "-force"])
        if not ok:
            res["status"] = "union_warp_fail"; return res
        union_dil = os.path.join(ws, "union_dil.mif")
        run(["maskfilter", union_nat, "dilate", "-npass", str(DILATE_VOX), union_dil, "-force"])
        fit_mask_mif = os.path.join(ws, "fit_mask.mif")
        ok, _, e = run(["mrcalc", union_dil, nmask, "-mult", fit_mask_mif, "-datatype", "bit", "-force"])
        if not ok:
            res["status"] = "fitmask_fail"; return res
        fit_mask_nii = os.path.join(ws, "fit_mask.nii.gz")
        run(["mrconvert", fit_mask_mif, fit_mask_nii, "-force"])

        data, affine = load_nifti(dwi_nii)
        data = data.astype(np.float32)
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=50)
        fmask, _ = load_nifti(fit_mask_nii)
        fmask = fmask > 0
        if fmask.sum() == 0:
            res["status"] = "empty_fitmask"; return res

        model = fwdti.FreeWaterTensorModel(gtab, fit_method=FIT_METHOD)
        fit = model.fit(data, mask=fmask)

        maps = {
            "MD": np.nan_to_num(fit.md).astype(np.float32),
            "FA": np.nan_to_num(fit.fa).astype(np.float32),
            "RD": np.nan_to_num(fit.rd).astype(np.float32),
            "AD": np.nan_to_num(fit.ad).astype(np.float32),
            "f":  np.clip(np.nan_to_num(fit.f), 0, 1).astype(np.float32),
        }
        for m, arr in maps.items():
            nat = os.path.join(ws, f"FW{m}_native.nii.gz")
            save_nifti(nat, arr, affine)
            ok, _, e = run(["mrtransform", nat, "-warp", fwd, "-reorient_fod", "no", tmpl_out[m], "-force"])
            if not ok:
                res["status"] = f"warp_fail_{m}"; return res

        res["status"] = "ok"
        return _extract(sid, tmpl_out, res)
    except Exception as exc:
        log.error(f"{sid}: CRASH {exc}\n{traceback.format_exc()[:400]}")
        res["status"] = "crash"; return res
    finally:
        for f in glob.glob(os.path.join(ws, "*")):
            try: os.remove(f)
            except Exception: pass
        try: os.rmdir(ws)
        except Exception: pass


def _extract(sid, tmpl_out, res):
    for m in FW_METRICS:
        img = tmpl_out[m]
        for net in NETWORKS:
            col = f"FW_{m}_{net}"
            nmask = os.path.join(VOXEL_MASK_DIR, f"{net}_voxel_mask.mif")
            if not (os.path.exists(img) and os.path.exists(nmask)):
                res[col] = np.nan; continue
            ok, out, _ = run(["mrstats", img, "-mask", nmask, "-ignorezero", "-output", "mean"])
            res[col] = float(out.strip().split()[0]) if (ok and out.strip()) else np.nan
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subject", type=str, default=None)
    args = ap.parse_args()

    meta = pd.read_csv(META_CSV, low_memory=False)
    meta["Subject"] = meta["Subject"].astype(str)
    sids = [args.subject] if args.subject else list(meta["Subject"])

    ws0 = os.path.join(WORKDIR, "fw_union")
    os.makedirs(ws0, exist_ok=True)
    union_template = build_template_union(ws0)
    if union_template is None:
        log.error("Could not build network union mask")
        return
    union_stable = os.path.join(FW_DIR, "net_union_template.mif")
    run(["mrconvert", union_template, union_stable, "-force"])

    if args.dry_run:
        log.info(f"union mask: {union_stable}")
        return

    rows = []
    if args.subject:
        rows.append(process_subject(sids[0], union_stable))
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(process_subject, sid, union_stable): sid for sid in sids}
            for fut in as_completed(futs):
                rows.append(fut.result())

    fw_df = pd.DataFrame(rows)
    keep = ["Subject"] + [c for c in fw_df.columns if c.startswith("FW_")]
    out = meta.merge(fw_df[keep], on="Subject", how="left")
    out.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
