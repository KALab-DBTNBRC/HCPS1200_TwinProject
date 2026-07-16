#!/usr/bin/env python3
"""
per_subject_SC_pipeline.py  —  HCP S1200 AUD twin study
========================================================
Generate one SIFT2-weighted 410×410 structural connectivity matrix per subject.
N=238 subjects. R740 build: 256 GB RAM, 80 cores, 126 GB /dev/shm.

Steps per subject
-----------------
  1. Warp composite atlas  (template → native)  [nearest-neighbour, single step]
  2. Sanitize atlas: clamp [0, 410], cast uint32
  3. Warp 5TT image        (template → native)  [linear]
  4. GMWMI seed mask from native 5TT
  5. 10M ACT tractography in native space
  6. SIFT2 weighting
  7. tck2connectome → 410×410 SIFT2-weighted, invnodevol-scaled, symmetric CSV
  8. Rescue outputs to disk; purge RAM disk

Resume behaviour
----------------
  Subjects with an existing connectome.npy are skipped automatically.
  Partial outputs from interrupted subjects are cleaned before retry.
  Safe to Ctrl-C and restart at any time.

Usage
-----
  python per_subject_SC_pipeline.py --dry-run         # verify all paths, no compute
  python per_subject_SC_pipeline.py --subject 191437  # single-subject test
  python per_subject_SC_pipeline.py                   # full batch
"""

import os, sys, shutil, logging, subprocess, time, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy  as np
import pandas as pd


#  CONFIGURATION 

BASE         = Path(os.environ.get("PROJECT_ROOT", "."))
PROCESSED    = BASE / "processed"       # .../processed/{MZ|DZ}/{PairXX}/sub-XXXXXX_{F|M}/
REGISTRATION = BASE / "registration"   # .../registration/sub-XXXXXX_warp_inv.mif
TEMPLATE_DIR = BASE / "study_template"
ATLAS_DIR    = BASE / "reference_atlas"
OUTPUT_DIR   = BASE / "SC_matrices"    # one sub-folder per subject
# NOTE: carries SSAGA/behavioral fields sourced from HCP Restricted-Access
# data. Never committed to this repository.
MASTER_XLSX  = Path(os.environ.get(
    "RESTRICTED_DERIVED_TWIN_TABLE",
    str(Path(os.environ.get("PROJECT_ROOT", ".")) / "twintables/Twins_240__all_vars.xlsx")
))

TEMPLATE_5TT = TEMPLATE_DIR / "5tt_template.mif"
ATLAS_TMPL   = ATLAS_DIR    / "Glasser_Tian_JHU_Composite_Template.nii.gz"

# compute settings
N_WORKERS    = 6    # parallel subjects  →  6 × 10 = 60 cores used
THREADS_PER  = 10   # MRtrix3 threads per subject
N_STREAMS    = "10M"

# storage settings
KEEP_TCK     = False  # TCK deleted after tck2connectome — saves ~2 TB
KEEP_WEIGHTS = True   # SIFT2 weights kept (238 × 40 MB ≈ 9.5 GB total)
RAM_BASE     = Path("/dev/shm")  # tmpfs for heavy intermediates

#  HELPERS 

def make_logger(name: str, log_file: Path) -> logging.Logger:
    """
    Subprocess-safe logger: writes only to per-subject file, never to stdout.
    propagate=False prevents bleed into the parent's root logger handlers.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
        )
        log.addHandler(fh)
    return log


def run(cmd: list, label: str, log: logging.Logger):
    """Run cmd; raise RuntimeError on non-zero exit; log stderr on failure."""
    log.info(f"[{label}]  " + " ".join(str(c) for c in cmd))
    t0   = time.time()
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error(f"FAILED ({elapsed:.0f}s)\n{proc.stderr.strip()}")
        raise RuntimeError(f"'{label}' exited {proc.returncode}")
    log.info(f"  → OK ({elapsed:.0f}s)")


def subject_dir(row: dict) -> Path:
    gender = "F" if int(row["Gender"]) == 0 else "M"
    return (
        PROCESSED
        / row["ZygosityGT1"]
        / row["TwinPairID"]
        / f"sub-{row['Subject']}_{gender}"
    )


def audit(csv_path: Path, log: logging.Logger) -> dict:
    A         = np.genfromtxt(csv_path, delimiter=",")
    n         = A.shape[0]
    density   = float(np.count_nonzero(A)) / max(n * (n - 1), 1)
    symmetric = bool(np.allclose(A, A.T, atol=1e-5))
    diag_zero = bool(np.all(np.diag(A) == 0))
    log.info(
        f"  Audit → shape={A.shape}  density={density:.4f}  "
        f"symmetric={symmetric}  diag_zero={diag_zero}  "
        f"min={A[A > 0].min():.4g}  max={A.max():.4g}"
    )
    if n != 410:
        log.warning(f"  *** Expected 410 nodes, got {n} — check atlas ***")
    return dict(
        nodes     = n,
        density   = round(density, 4),
        symmetric = symmetric,
        diag_zero = diag_zero,
    )


#  PER-SUBJECT PIPELINE 

def process_subject(row: dict) -> dict:
    sid    = str(row["Subject"])
    outdir = OUTPUT_DIR / f"sub-{sid}"
    outdir.mkdir(parents=True, exist_ok=True)

    log      = make_logger(sid, outdir / "pipeline.log")
    sep      = "─" * 60
    log.info(sep)
    log.info(f"sub-{sid}  |  {row['ZygosityGT1']}  {row['TwinPairID']}")

    # skip if already complete─
    final_npy = outdir / "connectome.npy"
    if final_npy.exists():
        log.info("connectome.npy exists — skipping")
        return dict(subject=sid, status="skipped")

    # clean partial outputs from any previous interrupted run─
    for leftover in ["connectome.csv", "sift2_weights.txt"]:
        p = outdir / leftover
        if p.exists():
            p.unlink()
            log.info(f"  Removed partial file: {leftover}")

    # resolve input paths
    subdir   = subject_dir(row)
    wmfod    = subdir / "wmfod_norm.mif"
    mask     = subdir / "mask.mif"
    inv_warp = REGISTRATION / f"sub-{sid}_warp_inv.mif"

    required = [
        (wmfod,        "wmfod_norm.mif"),
        (mask,         "mask.mif"),
        (inv_warp,     "warp_inv.mif"),
        (TEMPLATE_5TT, "5tt_template.mif"),
        (ATLAS_TMPL,   "atlas (template space)"),
    ]
    missing = [name for path, name in required if not path.exists()]
    if missing:
        log.error("Missing inputs: " + ", ".join(missing))
        return dict(subject=sid, status="missing_input",
                    detail=", ".join(missing))

    # RAM disk workspace for this subject
    ram = RAM_BASE / f"sub-{sid}"
    ram.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: warp atlas template → native─
        #    Single warp (atlas already in template space).
        #    -template wmfod ensures output matches native FOD grid exactly.
        #    No -assignment_radial_search (caused previous SIGSEGVs).
        atlas_float = ram / "atlas_float.mif"
        run(
            ["mrtransform", ATLAS_TMPL,
             "-warp",     inv_warp,
             "-interp",   "nearest",
             "-template", wmfod,
             "-datatype", "float32",
             atlas_float, "-force",
             "-nthreads", THREADS_PER],
            "atlas → native", log,
        )

        # Step 2: sanitize → clamp [0, 410] → cast uint32─
        atlas_sane = ram / "atlas_sane.mif"
        run(
            ["mrcalc", atlas_float, "0", "-max", "410", "-min",
             atlas_sane, "-force", "-datatype", "float32"],
            "atlas clamp [0, 410]", log,
        )
        atlas_int = ram / "atlas_uint32.mif"
        run(
            ["mrconvert", atlas_sane, atlas_int,
             "-datatype", "uint32", "-force"],
            "atlas cast uint32", log,
        )

        # Step 3: warp 5TT template → native
        tt_native = ram / "5tt_native.mif"
        run(
            ["mrtransform", TEMPLATE_5TT,
             "-warp",     inv_warp,
             "-template", wmfod,
             tt_native, "-force",
             "-nthreads", THREADS_PER],
            "5TT → native", log,
        )

        # Step 4: GMWMI seed mask
        gmwmi = ram / "gmwmi.mif"
        run(["5tt2gmwmi", tt_native, gmwmi, "-force"],
            "GMWMI", log)

        # Step 5: 10M ACT tractography
        tck = ram / "tracks_10M.tck"
        run(
            ["tckgen", wmfod, tck,
             "-act",          tt_native,
             "-seed_gmwmi",   gmwmi,
             "-select",       N_STREAMS,
             "-minlength",    "10",
             "-maxlength",    "250",
             "-cutoff",       "0.06",
             "-backtrack",
             "-crop_at_gmwmi",
             "-nthreads",     THREADS_PER,
             "-force"],
            "10M ACT tractography", log,
        )

        # Step 6: SIFT2 weighting
        weights = ram / "sift2_weights.txt"
        run(
            ["tcksift2", tck, wmfod, weights,
             "-nthreads", THREADS_PER, "-force"],
            "SIFT2", log,
        )

        # Rescue weights before connectome step
        weights_out = outdir / "sift2_weights.txt"
        if KEEP_WEIGHTS:
            shutil.copy(weights, weights_out)
            log.info("  SIFT2 weights rescued")

        if KEEP_TCK:
            shutil.copy(tck, outdir / "tracks_10M.tck")
            log.info("  TCK rescued")

        # Step 7: tck2connectome─
        csv_tmp = ram / "connectome.csv"
        
        run(
            ["tck2connectome", tck, atlas_int, csv_tmp,
             "-tck_weights_in",       weights,
             "-scale_invnodevol",
             "-symmetric",
             "-zero_diagonal",
             "-assignment_radial_search", "3",
             "-nthreads",               THREADS_PER,
             "-force"],
            "tck2connectome", log,
        )

        # Step 8: rescue CSV + NPY─
        csv_out = outdir / "connectome.csv"
        shutil.copy(csv_tmp, csv_out)

        A = np.genfromtxt(csv_out, delimiter=",")
        np.save(final_npy, A)

        log.info(f"  Saved: {csv_out}")
        log.info(f"  Saved: {final_npy}")

        stats = audit(csv_out, log)
        log.info(f"sub-{sid}: COMPLETE")
        return dict(subject=sid, status="ok", **stats)

    except Exception as exc:
        log.error(f"sub-{sid}: FAILED — {exc}")
        return dict(subject=sid, status="failed", detail=str(exc))

    finally:
        if ram.exists():
            shutil.rmtree(ram, ignore_errors=True)
            log.info(f"  RAM disk purged: {ram}")


#  MAIN 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true",
                        help="Verify all input paths; no compute")
    parser.add_argument("--subject",  type=str, default=None,
                        help="Single HCP subject ID (e.g. 191437)")
    args = parser.parse_args()

    # master log (parent process only)─
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    root    = logging.getLogger("main")
    root.setLevel(logging.INFO)
    root.propagate = False
    for h in [logging.StreamHandler(sys.stdout),
               logging.FileHandler(OUTPUT_DIR / "SC_pipeline_master.log", mode="a")]:
        h.setFormatter(log_fmt)
        root.addHandler(h)

    # load master table─
    if not MASTER_XLSX.exists():
        root.error(f"Master table not found: {MASTER_XLSX}")
        sys.exit(1)

    df = pd.read_excel(MASTER_XLSX, engine="openpyxl")
    df["Subject"] = df["Subject"].astype(str)
    root.info(f"Loaded {len(df)} subjects")

    if args.subject:
        df = df[df["Subject"] == args.subject]
        if df.empty:
            root.error(f"Subject {args.subject} not in master table")
            sys.exit(1)

    rows = df.to_dict(orient="records")

    # dry-run─
    if args.dry_run:
        root.info("DRY RUN — checking inputs only")
        problems = []
        for row in rows:
            sid      = row["Subject"]
            sdir     = subject_dir(row)
            inv_warp = REGISTRATION / f"sub-{sid}_warp_inv.mif"
            missing  = [n for p, n in [
                (sdir / "wmfod_norm.mif", "wmfod_norm"),
                (sdir / "mask.mif",       "mask"),
                (inv_warp,                "warp_inv"),
            ] if not p.exists()]
            if missing:
                problems.append(sid)
                root.warning(f"  sub-{sid}: MISSING {missing}")
            else:
                root.info(f"  sub-{sid}: OK")

        # shared files
        for p, n in [(TEMPLATE_5TT, "5tt_template"), (ATLAS_TMPL, "atlas")]:
            if not p.exists():
                root.error(f"MISSING shared file: {n} ({p})")

        already_done = sum(
            1 for r in rows
            if (OUTPUT_DIR / f"sub-{r['Subject']}" / "connectome.npy").exists()
        )
        root.info(
            f"\nDry-run summary: {len(rows)} subjects total | "
            f"{already_done} already done | "
            f"{len(problems)} with missing inputs"
        )
        return

    # single-subject or full batch
    n_workers = 1 if args.subject else N_WORKERS
    already   = sum(
        1 for r in rows
        if (OUTPUT_DIR / f"sub-{r['Subject']}" / "connectome.npy").exists()
    )
    to_run = len(rows) - already

    root.info(
        f"\nStarting: {len(rows)} total | {already} skipped (done) | "
        f"{to_run} to process | {n_workers} workers | "
        f"{THREADS_PER} threads/subject"
    )

    if n_workers == 1:
        results = [process_subject(rows[0])]
    else:
        results = []
        with Pool(processes=n_workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(process_subject, rows), start=1
            ):
                results.append(res)
                status = res.get("status", "?")
                root.info(
                    f"  [{i}/{len(rows)}] sub-{res['subject']} → {status}"
                    + (f"  density={res['density']}" if status == "ok" else "")
                )

    # summary─
    rdf     = pd.DataFrame(results)
    summary = rdf["status"].value_counts().to_dict()
    root.info(f"\nPipeline complete: {summary}")

    report = OUTPUT_DIR / "SC_pipeline_report.csv"
    rdf.to_csv(report, index=False)
    root.info(f"Report: {report}")

    failed = rdf[rdf["status"] == "failed"]
    if len(failed):
        root.warning(
            f"\n{len(failed)} FAILED:\n"
            + "\n".join(
                f"  sub-{r['subject']}: {r.get('detail', '')}"
                for _, r in failed.iterrows()
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()