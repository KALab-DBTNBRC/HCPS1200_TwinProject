#!/usr/bin/env python3
"""
phaseB2_noddi_isovf.py  --  HCP S1200 AUD twin study
===================================================
Why this exists
---------------
Phase B (DIPY fwdti) FAILED a coherence check: the free-water-corrected MD was
nearly uncorrelated with standard MD (R^2~0.04), and the free-water fraction f
ran NEGATIVE with age (real free water rises with age). The single-voxel bi-tensor
free-water-elimination problem is ill-posed without spatial regularization, so that
fit could not partition the signal. The free-water hypothesis is therefore still
UNTESTED, not resolved.

NODDI's isotropic volume fraction (ISOVF) is a constrained, multi-compartment
estimate of the free-water / CSF-like compartment that uses the full multi-shell
data and is far better posed. AMICO makes it fast (precomputed kernels).

This script, per subject, mirrors Phase B exactly:
  1. export dwi_biascorr.mif -> NIfTI + FSL bvecs/bvals + AMICO scheme
  2. fit NODDI (AMICO) within the brain mask  -> ISOVF, ICVF, OD
  3. warp ISOVF/ICVF/OD native -> template with sub-{sid}_warp_fwd.mif (scalar)
  4. mean within each network voxel mask  (phase6Avoxelmetrics.py pattern)

Output: BASE/final_stats/MASTER_NODDI.csv  (merges on Subject)
Columns: ISOVF_{net}, ICVF_{net}, OD_{net}

** BUILT-IN VALIDITY GATE **
After fitting, the script correlates ISOVF with Age. Real free water INCREASES with
age. If corr(ISOVF, Age) is POSITIVE -> the estimate is trustworthy, proceed to the
association test. If NEGATIVE -> ISOVF is also degenerate; do NOT interpret, stop and
write the localization paper. The gate prints GO / NO-GO automatically.

Dependencies (conda env dtiproject):
  pip install dmri-amico            # provides `import amico`
  (MRtrix3, numpy, pandas, scipy already present)

HCP shares one gradient scheme across subjects, so the slow kernel atoms are built
ONCE and reused (shared ATOMS_path) -> fast batch.

Usage
-----
  python phaseB2_noddi_isovf.py --dry-run
  python phaseB2_noddi_isovf.py --subject 191437     # test one first (catch API issues)
  python phaseB2_noddi_isovf.py                       # full batch (resumable)
  python phaseB2_noddi_isovf.py --gate-only           # just recompute the age gate from CSV
"""

import os, glob, argparse, subprocess, logging, traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")     # AMICO is internally threaded; keep modest

# -- PATHS (identical to Phase B) --
BASE          = os.environ.get("PROJECT_ROOT", ".")
PROCESSED_DIR = os.path.join(BASE, "processed")
REG_DIR       = os.path.join(BASE, "registration")
VOXEL_MASK_DIR= os.path.join(BASE, "final_masks", "voxel")
NODDI_DIR     = os.path.join(BASE, "noddi")
META_CSV      = os.path.join(BASE, "twintables", "network_roi_metrics_FINAL.csv")
OUT_CSV       = os.path.join(BASE, "final_stats", "MASTER_NODDI.csv")
SHARED_ATOMS  = os.path.join(NODDI_DIR, "shared_kernels")   # build NODDI atoms once

NETWORKS      = ["Reward", "Salience", "DMN", "Olfactory"]
METRICS       = ["ISOVF", "ICVF", "OD"]      # AMICO writes FIT_ISOVF/FIT_ICVF/FIT_OD
N_WORKERS     = 6                             # AMICO is threaded; 6 keeps RAM/threads sane
WORKDIR       = "/dev/shm"

os.makedirs(NODDI_DIR, exist_ok=True)
os.makedirs(os.path.join(NODDI_DIR, "template"), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
LOG_FILE = os.path.join(NODDI_DIR, f"phaseB2_noddi_{datetime.now():%Y%m%d_%H%M}.log")
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


def fit_one(sid):
    res = {"Subject": str(sid), "status": "?"}
    sdir = find_subject_dir(sid)
    if sdir is None:
        res["status"] = "no_dir"; return res
    dwi  = os.path.join(sdir, "dwi_biascorr.mif")
    nmask= os.path.join(sdir, "mask.mif")
    fwd  = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
    for p in (dwi, nmask, fwd):
        if not os.path.exists(p):
            res["status"] = f"missing:{os.path.basename(p)}"; return res

    tmpl_out = {m: os.path.join(NODDI_DIR, "template", f"sub-{sid}_{m}_tmpl.mif")
                for m in METRICS}
    if all(os.path.exists(p) for p in tmpl_out.values()):
        res["status"] = "exists"; return _extract(sid, tmpl_out, res)

    ws = os.path.join(WORKDIR, f"noddi_{sid}")
    subj = "subj"
    sdir_amico = os.path.join(ws, subj)
    os.makedirs(sdir_amico, exist_ok=True)
    try:
        import amico

        # 1. export DWI + grads, build AMICO scheme
        dwi_nii = os.path.join(sdir_amico, "dwi.nii.gz")
        fbval   = os.path.join(sdir_amico, "bvals")
        fbvec   = os.path.join(sdir_amico, "bvecs")
        mask_nii= os.path.join(sdir_amico, "mask.nii.gz")
        ok, _, e = run(["mrconvert", dwi, dwi_nii,
                        "-export_grad_fsl", fbvec, fbval, "-force"])
        if not ok:
            res["status"] = "mrconvert_dwi"; log.warning(f"{sid}: {e[:160]}"); return res
        run(["mrconvert", nmask, mask_nii, "-force"])
        scheme = os.path.join(sdir_amico, "noddi.scheme")
        amico.util.fsl2scheme(fbval, fbvec, scheme)

        # 2. AMICO NODDI fit (reuse shared atoms; build them once if absent)
        amico.setup()
        ae = amico.Evaluation(study_path=ws, subject=subj)
        ae.set_config("ATOMS_path", SHARED_ATOMS)
        ae.set_config("OUTPUT_path", os.path.join(sdir_amico, "AMICO", "NODDI"))
        ae.load_data(dwi_filename=dwi_nii, scheme_filename=scheme,
                     mask_filename=mask_nii, b0_thr=50)
        ae.set_model("NODDI")
        first = not os.path.isdir(SHARED_ATOMS) or not os.listdir(SHARED_ATOMS)
        ae.generate_kernels(regenerate=first)   # atoms depend only on the scheme
        ae.load_kernels()
        ae.fit()
        ae.save_results()

        outdir = os.path.join(sdir_amico, "AMICO", "NODDI")
        fmap = {"ISOVF": "FIT_ISOVF.nii.gz", "ICVF": "FIT_ICVF.nii.gz", "OD": "FIT_OD.nii.gz"}

        # 3. warp each native map -> template (scalar)
        for m, fn in fmap.items():
            native = os.path.join(outdir, fn)
            if not os.path.exists(native):
                # some AMICO versions write without .gz
                alt = native[:-3]
                native = alt if os.path.exists(alt) else native
            if not os.path.exists(native):
                res["status"] = f"no_{m}"; log.warning(f"{sid}: missing {fn}"); return res
            ok, _, e = run(["mrtransform", native, "-warp", fwd,
                            "-reorient_fod", "no", tmpl_out[m], "-force"])
            if not ok:
                res["status"] = f"warp_{m}"; log.warning(f"{sid}: {e[:160]}"); return res

        res["status"] = "ok"
        return _extract(sid, tmpl_out, res)

    except Exception as exc:
        log.error(f"{sid}: CRASH {exc}\n{traceback.format_exc()[:500]}")
        res["status"] = "crash"; return res
    finally:
        for root, _, files in os.walk(ws, topdown=False):
            for f in files:
                try: os.remove(os.path.join(root, f))
                except Exception: pass
            try: os.rmdir(root)
            except Exception: pass


def _extract(sid, tmpl_out, res):
    for m in METRICS:
        img = tmpl_out[m]
        for net in NETWORKS:
            col = f"{m}_{net}"
            nmask = os.path.join(VOXEL_MASK_DIR, f"{net}_voxel_mask.mif")
            if not (os.path.exists(img) and os.path.exists(nmask)):
                res[col] = np.nan; continue
            ok, out, _ = run(["mrstats", img, "-mask", nmask,
                              "-ignorezero", "-output", "mean"])
            res[col] = float(out.strip().split()[0]) if (ok and out.strip()) else np.nan
    return res


def age_gate(df, meta):
    """Validity check: real free water (ISOVF) rises with age. Print GO/NO-GO."""
    m = meta[["Subject", "Age_in_Yrs"]].copy()
    m["Subject"] = m["Subject"].astype(str)
    g = df.merge(m, on="Subject", how="left").dropna(subset=["Age_in_Yrs"])
    log.info("\n" + "=" * 60)
    log.info("VALIDITY GATE -- ISOVF vs Age (must be POSITIVE to trust)")
    log.info("=" * 60)
    verdict = []
    for net in NETWORKS:
        col = f"ISOVF_{net}"
        if col not in g or g[col].notna().sum() < 20:
            continue
        sub = g.dropna(subset=[col])
        r, p = stats.pearsonr(sub[col], sub["Age_in_Yrs"])
        tag = "OK (+)" if r > 0 else "FAIL (-)"
        log.info(f"  {col:16s} r(Age) = {r:+.3f}  p = {p:.3g}   {tag}")
        verdict.append(r > 0)
    if verdict and all(verdict):
        log.info("  --> GO: ISOVF rises with age in all networks; estimate is trustworthy.")
        log.info("      Proceed to the between-family association + absorption test.")
    elif verdict and not any(verdict):
        log.info("  --> NO-GO: ISOVF FALLS with age (like fwdti's f). Degenerate.")
        log.info("      Do NOT interpret. Write the family-level localization paper.")
    else:
        log.info("  --> MIXED: inspect per network before interpreting.")
    log.info("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    meta = pd.read_csv(META_CSV, low_memory=False)
    meta["Subject"] = meta["Subject"].astype(str)

    if args.gate_only:
        df = pd.read_csv(OUT_CSV); df["Subject"] = df["Subject"].astype(str)
        age_gate(df, meta); return

    sids = [args.subject] if args.subject else list(meta["Subject"])

    if args.dry_run:
        n_ok = sum(bool(find_subject_dir(s)
                        and os.path.exists(os.path.join(REG_DIR, f"sub-{s}_warp_fwd.mif")))
                   for s in sids)
        try:
            import amico; have = "yes"
        except Exception:
            have = "NO  (pip install dmri-amico)"
        log.info(f"amico importable: {have}")
        log.info(f"Dry-run: {n_ok}/{len(sids)} subjects ready; workers={N_WORKERS}")
        return

    log.info(f"AMICO-NODDI on {len(sids)} subjects ({N_WORKERS} workers)")
    rows = []
    if args.subject:
        rows.append(fit_one(sids[0]))           # serial: builds shared atoms, easy debug
    else:
        # build shared kernels once on the first subject (serial), then parallelise
        log.info("Building shared NODDI kernels on first subject (one-time)...")
        rows.append(fit_one(sids[0]))
        rest = sids[1:]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(fit_one, s): s for s in rest}
            done = 1
            for fut in as_completed(futs):
                r = fut.result(); rows.append(r); done += 1
                if r["status"] not in ("ok", "exists"):
                    log.warning(f"  sub-{r['Subject']}: {r['status']}")
                if done % 20 == 0:
                    log.info(f"  {done}/{len(sids)} done")

    df = pd.DataFrame(rows)
    log.info(f"Status: {df['status'].value_counts().to_dict()}")
    keep = ["Subject"] + [f"{m}_{n}" for m in METRICS for n in NETWORKS]
    out = meta.merge(df[[c for c in keep if c in df.columns]], on="Subject", how="left")
    out.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_CSV}  ({len(out)} rows)")

    age_gate(df, meta)
    log.info("Next (only if GO): between-family drinking -> ISOVF, and test whether "
             "familial ISOVF absorbs the between-family standard-MD beta (target 0.230).")


if __name__ == "__main__":
    main()
