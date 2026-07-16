"""
06_extract_metrics.py

Corresponds to Methods: fixel metrics (FD/FC/FDC) and conventional tensor (FA/MD/RD/AD) extraction. Three levels, fused because all extract per-subject scalar metrics into the same master ROI table.
"""

import os
import pandas as pd
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor
import glob
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# LEVEL 1 -- fibre density (FD) fixel metric extraction (orig. phase6Afixelmetrics.py)

# CONFIGURATION
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
# NOTE: this derived table carries SSAGA/behavioral fields sourced from
# HCP Restricted-Access data. Never committed to this repository.
CSV_PATH = os.environ.get(
    "RESTRICTED_DERIVED_TWIN_TABLE",
    os.path.join(BASE_DIR, 'twintables/Twins240_DTI - Sheet1.csv')
)
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')
TEMPLATE_MASK = os.path.join(TEMPLATE_DIR, 'template_fixel_mask')

# Paths for inputs
REG_DIR = os.path.join(BASE_DIR, 'registration')
FOD_INPUT_DIR = os.path.join(BASE_DIR, 'processed/template_inputs/fods')

# Output Metrics Directories
METRICS_BASE = os.path.join(BASE_DIR, 'fixel_metrics')
FD_DIR = os.path.join(METRICS_BASE, 'fd')
FC_DIR = os.path.join(METRICS_BASE, 'fc')
FDC_DIR = os.path.join(METRICS_BASE, 'fdc')

for d in [FD_DIR, FC_DIR, FDC_DIR]:
    os.makedirs(d, exist_ok=True)

# Parallel Config
MAX_WORKERS = 8 
THREADS_PER_SUBJECT = 8 

def process_single_subject(subject_info):
    sid, sex_char = subject_info
    filename = f"sub-{sid}_{sex_char}.mif"
    
    subj_fod = os.path.join(FOD_INPUT_DIR, filename)
    warp_fwd = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
    warped_fod = os.path.join(METRICS_BASE, f"sub-{sid}_fod_warped.mif")
    subj_fd = os.path.join(FD_DIR, f"sub-{sid}.mif")
    subj_fc = os.path.join(FC_DIR, f"sub-{sid}.mif")
    subj_fdc = os.path.join(FDC_DIR, f"sub-{sid}.mif")
    tmp_fixel_dir = os.path.join(METRICS_BASE, f"tmp_{sid}")

    if not os.path.exists(warp_fwd):
        return f"MISSING WARP: {sid}"
    
    try:
        # 1. Warp & Reorient
        subprocess.run(["mrtransform", subj_fod, "-warp", warp_fwd, 
                        "-reorient_fod", "yes", warped_fod, "-nthreads", str(THREADS_PER_SUBJECT), "-force"], check=True)

        # 2. Extract FD
        os.makedirs(tmp_fixel_dir, exist_ok=True)
        subprocess.run(["fod2fixel", warped_fod, tmp_fixel_dir, "-fmls_peak_value", "0.06", "-force"], check=True)
        subprocess.run(["fixelcorrespondence", tmp_fixel_dir, TEMPLATE_MASK, FD_DIR, f"sub-{sid}.mif"], check=True)

        # 3. Extract FC
        subprocess.run(["warp2metric", warp_fwd, "-fc", TEMPLATE_MASK, FC_DIR, f"sub-{sid}.mif"], check=True)

        # 4. Compute FDC
        subprocess.run(["mrcalc", subj_fd, subj_fc, "-mult", subj_fdc, "-force"], check=True)

        # Clean up intermediate files
        if os.path.exists(warped_fod): os.remove(warped_fod)
        subprocess.run(["rm", "-rf", tmp_fixel_dir])
        
        return f"SUCCESS: {sid}"
    
    except Exception as e:
        return f"ERROR: {sid} - {str(e)}"

def run_parallel_extraction():
    df = pd.read_csv(CSV_PATH)
    gender_map = {1: 'M', 0: 'F'}
    subject_list = [(str(int(row['Subject'])), gender_map.get(int(row['Gender']))) for _, row in df.iterrows()]

    print(f"Starting metric extraction for {len(subject_list)} subjects...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_subject, subject_list))

    for res in results: print(res)

def run_level_1():
    run_parallel_extraction()

# LEVEL 2 -- conventional tensor (DTI) voxel metric extraction -- FA/MD/RD/AD (orig. phase6Avoxelmetrics.py)

#!/usr/bin/env python3
"""
dti_metrics_final.py
====================
Two-step script:

STEP 1 — Rerun tensor2metric with -md (not -adc) for all 238 subjects.
          Tensor files already exist so this is fast (~30 min total).

STEP 2 — Extract FA/MD/RD/AD mean values from network voxel masks
          and append to network_roi_metrics_FINAL.csv.

Fixes from previous script:
  - -adc replaced with -md
  - -nthreads removed from tensor2metric (unsupported flag)
  - template dimension verification after first subject
  - clean output to MASTER_ROI_METRICS_DTI_FBA.csv
"""


#  PATHS 
BASE_DIR      = os.environ.get("PROJECT_ROOT", ".")
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
REG_DIR       = os.path.join(BASE_DIR, 'registration')
DTI_OUT_DIR   = os.path.join(BASE_DIR, 'dti_metrics')
TEMPLATE_DIR  = os.path.join(BASE_DIR, 'study_template')

#  YOUR FINAL NETWORK MASKS 
# Voxel-space binary masks (3D .mif) — used for DTI scalar extraction
# These are the outputs of your phase8networkmaskmaker.py,
# converted to voxel space via fixel2voxel or direct atlas registration.
# Update this path to wherever you regenerate your clean masks.
VOXEL_MASK_DIR = os.path.join(BASE_DIR, 'final_masks', 'voxel')

# Network names must match mask filenames: {NET}_voxel_mask.mif
NETWORKS = ['Reward', 'Salience', 'DMN', 'Olfactory']
DTI_METRICS = ['fa', 'md', 'rd', 'ad']

#  INPUT METADATA 
META_CSV = os.path.join(BASE_DIR, 'twintables', 'network_roi_metrics_FINAL.csv')
OUT_CSV  = os.path.join(BASE_DIR, 'final_stats', 'MASTER_ROI_METRICS_DTI_FBA.csv')

#  PARALLELISM ─
MAX_WORKERS       = 15   # 15 workers × 3 threads = 45 cores
THREADS_PER_CMD   = "3"  # for dwi2tensor and mrtransform only

#  LOGGING ─
os.makedirs(os.path.join(DTI_OUT_DIR, 'tensors'), exist_ok=True)
for m in DTI_METRICS:
    os.makedirs(os.path.join(DTI_OUT_DIR, 'native', m), exist_ok=True)
    os.makedirs(os.path.join(DTI_OUT_DIR, 'warped', m), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

LOG_FILE = os.path.join(DTI_OUT_DIR,
    f"dti_final_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger()


#  UTILITIES ─
def run(cmd, use_threads=False, use_force=True):
    """Run command. Only add -nthreads to commands that support it."""
    if use_threads:
        cmd += ['-nthreads', THREADS_PER_CMD]
    if use_force:
        cmd += ['-force']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def get_dims(mif_path):
    """Return (x, y, z) voxel dimensions of a .mif file."""
    result = subprocess.run(['mrinfo', mif_path, '-size'],
                            capture_output=True, text=True)
    parts = result.stdout.strip().split()
    if len(parts) >= 3:
        return tuple(int(p) for p in parts[:3])
    return None


#  STEP 1: RERUN TENSOR2METRIC WITH -md 
def build_dti_metrics(sub_dir):
    """Per-subject: fit tensor → extract FA/MD/RD/AD → warp to template."""
    folder  = os.path.basename(sub_dir)
    sid     = folder.split('_')[0].replace('sub-', '')

    dwi     = os.path.join(sub_dir, 'dwi_biascorr.mif')
    mask    = os.path.join(sub_dir, 'mask.mif')
    warp    = os.path.join(REG_DIR, f'sub-{sid}_warp_fwd.mif')
    tensor  = os.path.join(DTI_OUT_DIR, 'tensors', f'sub-{sid}_tensor.mif')

    if not os.path.exists(dwi) or not os.path.exists(warp):
        return sid, 'SKIP', 'Missing DWI or warp'

    native = {m: os.path.join(DTI_OUT_DIR, 'native', m, f'sub-{sid}_{m}.mif')
              for m in DTI_METRICS}
    warped = {m: os.path.join(DTI_OUT_DIR, 'warped', m, f'sub-{sid}_{m}_warped.mif')
              for m in DTI_METRICS}

    try:
        # Step A: Fit tensor (skip if exists)
        if not os.path.exists(tensor):
            ok, err = run(['dwi2tensor', dwi, tensor, '-mask', mask],
                          use_threads=True)
            if not ok:
                return sid, 'FAIL', f'dwi2tensor: {err[:200]}'

        # Step B: Extract metrics
        # NOTE: -nthreads NOT passed to tensor2metric (unsupported)
        # NOTE: -md for mean diffusivity (NOT -adc)
        need_regen = not os.path.exists(native['fa']) or not os.path.exists(native['md'])
        if need_regen:
            ok, err = run([
                'tensor2metric', tensor,
                '-fa', native['fa'],
                '-adc', native['md'],
                '-rd', native['rd'],
                '-ad', native['ad'],
                '-mask', mask
            ], use_threads=False)       # ← tensor2metric does NOT take -nthreads
            if not ok:
                return sid, 'FAIL', f'tensor2metric: {err[:200]}'

        # Step C: Warp to template space
        for m in DTI_METRICS:
            if not os.path.exists(warped[m]):
                ok, err = run([
                    'mrtransform', native[m],
                    '-warp', warp,
                    warped[m],
                    '-reorient_fod', 'no'   # correct for scalar images
                ], use_threads=True)
                if not ok:
                    return sid, 'FAIL', f'mrtransform {m}: {err[:200]}'

        return sid, 'OK', ''

    except Exception as e:
        return sid, 'CRASH', str(e)


def run_dti_pipeline():
    log.info("=== STEP 1: DTI TENSOR → METRICS → WARP ===")
    subject_dirs = sorted(glob.glob(
        os.path.join(PROCESSED_DIR, '*', '*', 'sub-*')))
    log.info(f"  {len(subject_dirs)} subjects queued")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(build_dti_metrics, d): d for d in subject_dirs}
        for fut in as_completed(futures):
            sid, status, msg = fut.result()
            results.append({'Subject': sid, 'Status': status, 'Note': msg})
            if status != 'OK':
                log.warning(f"  {sid}: {status} — {msg[:80]}")

    ok   = sum(1 for r in results if r['Status'] == 'OK')
    skip = sum(1 for r in results if r['Status'] == 'SKIP')
    fail = sum(1 for r in results if r['Status'] not in ('OK', 'SKIP'))
    log.info(f"  Done: {ok} OK | {skip} skipped | {fail} failed")

    # Verify one subject's warped FA is in template space
    sample_fa = glob.glob(os.path.join(DTI_OUT_DIR, 'warped', 'fa', '*.mif'))
    if sample_fa:
        dims = get_dims(sample_fa[0])
        expected = (137, 176, 139)
        if dims == expected:
            log.info(f"  Template space verified: {dims} ✓")
        else:
            log.warning(f"  Template space MISMATCH: got {dims}, expected {expected}")
            log.warning("  Check mrtransform warp direction")

    return pd.DataFrame(results)


#  STEP 2: EXTRACT DTI ROI MEANS ─
def extract_subject_dti(sid):
    """Extract mean FA/MD/RD/AD per network for one subject."""
    row = {'Subject': str(sid)}
    for m in DTI_METRICS:
        img = os.path.join(DTI_OUT_DIR, 'warped', m, f'sub-{sid}_{m}_warped.mif')
        if not os.path.exists(img):
            for net in NETWORKS:
                row[f'{m.upper()}_{net}'] = np.nan
            continue
        for net in NETWORKS:
            mask = os.path.join(VOXEL_MASK_DIR, f'{net}_voxel_mask.mif')
            if not os.path.exists(mask):
                row[f'{m.upper()}_{net}'] = np.nan
                continue
            cmd = ['mrstats', img, '-mask', mask, '-ignorezero', '-output', 'mean']
            result = subprocess.run(cmd, capture_output=True, text=True)
            val = result.stdout.strip()
            row[f'{m.upper()}_{net}'] = float(val) if val else np.nan
    return row


def run_dti_extraction():
    log.info("=== STEP 2: EXTRACT DTI ROI MEANS ===")

    # Check masks exist
    missing_masks = []
    for net in NETWORKS:
        mask = os.path.join(VOXEL_MASK_DIR, f'{net}_voxel_mask.mif')
        if not os.path.exists(mask):
            missing_masks.append(mask)
    if missing_masks:
        log.error("MISSING VOXEL MASKS — run your networkmasker first:")
        for m in missing_masks:
            log.error(f"  {m}")
        return None

    # Get subject list from existing warped FA files
    fa_files = sorted(glob.glob(
        os.path.join(DTI_OUT_DIR, 'warped', 'fa', '*.mif')))
    sids = [os.path.basename(f).replace('sub-', '').replace('_fa_warped.mif', '')
            for f in fa_files]
    log.info(f"  Extracting DTI metrics for {len(sids)} subjects...")

    all_rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(extract_subject_dti, sid): sid for sid in sids}
        done = 0
        for fut in as_completed(futures):
            all_rows.append(fut.result())
            done += 1
            if done % 50 == 0:
                log.info(f"  {done}/{len(sids)} subjects extracted")

    dti_df = pd.DataFrame(all_rows)
    log.info(f"  Extraction complete: {len(dti_df)} subjects")
    return dti_df


#  STEP 3: MERGE WITH MASTER TABLE ─
def merge_and_save(dti_df):
    log.info("=== STEP 3: MERGE WITH MASTER TABLE ===")

    meta_df = pd.read_csv(META_CSV, low_memory=False)
    meta_df['Subject'] = meta_df['Subject'].astype(str)
    dti_df['Subject']  = dti_df['Subject'].astype(str)

    # Check overlap
    meta_subs = set(meta_df['Subject'])
    dti_subs  = set(dti_df['Subject'])
    overlap   = meta_subs & dti_subs
    log.info(f"  Meta: {len(meta_subs)} | DTI: {len(dti_subs)} | Overlap: {len(overlap)}")

    final_df = pd.merge(meta_df, dti_df, on='Subject', how='inner')

    if len(final_df) == 0:
        log.error("MERGE FAILED: 0 rows. Check subject ID format.")
        return

    final_df.to_csv(OUT_CSV, index=False)
    log.info(f"  Saved: {OUT_CSV}")
    log.info(f"  Rows: {len(final_df)} | Columns: {len(final_df.columns)}")

    # Quick QC: check for missing DTI values
    dti_cols = [f'{m.upper()}_{net}'
                for m in DTI_METRICS for net in NETWORKS]
    missing  = final_df[dti_cols].isna().sum()
    if missing.sum() > 0:
        log.warning("  Missing values in DTI columns:")
        for col, n in missing[missing > 0].items():
            log.warning(f"    {col}: {n} missing")
    else:
        log.info("  All DTI columns complete — no missing values")


#  ENTRY POINT ─

def run_level_2():
    log.info("=" * 60)
    log.info("DTI METRICS PIPELINE — FINAL")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Step 1: Build DTI metrics (skips subjects already done)
    report_df = run_dti_pipeline()
    report_df.to_csv(
        os.path.join(DTI_OUT_DIR, 'dti_pipeline_report.csv'), index=False)

    # Step 2: Extract ROI means
    dti_df = run_dti_extraction()
    if dti_df is None:
        log.error("Extraction failed — check mask paths above")
    else:
        # Step 3: Merge and save
        merge_and_save(dti_df)

    log.info(f"\nLog saved to: {LOG_FILE}")

# LEVEL 3 -- fibre cross-section (FC) / FDC extraction, with rescue pass for incomplete subjects (orig. phase6Bfdcextractor.py)

#CONFIGURATION
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
# NOTE: this derived table carries SSAGA/behavioral fields sourced from
# HCP Restricted-Access data. Never committed to this repository.
CSV_PATH = os.environ.get(
    "RESTRICTED_DERIVED_TWIN_TABLE",
    os.path.join(BASE_DIR, 'twintables/Twins240_DTI - Sheet1.csv')
)
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')
VOXEL_MASK = os.path.join(TEMPLATE_DIR, 'template_mask.mif') 
TEMPLATE_FIXEL_MASK = os.path.join(TEMPLATE_DIR, 'template_fixel_mask')

METRICS_BASE = os.path.join(BASE_DIR, 'fixel_metrics')
FD_DIR = os.path.join(METRICS_BASE, 'fd')
FC_DIR = os.path.join(METRICS_BASE, 'fc')
FDC_DIR = os.path.join(METRICS_BASE, 'fdc')
REG_DIR = os.path.join(BASE_DIR, 'registration')

MAX_WORKERS = 12 

def super_rescue(subject_info):
    sid, _ = subject_info
    
    warped_fod = os.path.join(METRICS_BASE, f"sub-{sid}_fod_warped.mif")
    warp_fwd = os.path.join(REG_DIR, f"sub-{sid}_warp_fwd.mif")
    tmp_fixel_dir = os.path.join(METRICS_BASE, f"tmp_rescue_{sid}")
    
    subject_fd_file = os.path.join(tmp_fixel_dir, "fd.mif")
    
    subj_fd_out = os.path.join(FD_DIR, f"sub-{sid}.mif")
    subj_fc_out = os.path.join(FC_DIR, f"sub-{sid}.mif")
    subj_fdc_out = os.path.join(FDC_DIR, f"sub-{sid}.mif")

    if not os.path.exists(warped_fod):
        return f"SKIP: {sid} (Warped FOD missing)"

    try:
        # Re-segment
        os.makedirs(tmp_fixel_dir, exist_ok=True)
        subprocess.run(["fod2fixel", warped_fod, tmp_fixel_dir, 
                        "-afd", "fd.mif",
                        "-mask", VOXEL_MASK, "-fmls_peak_value", "0.06", "-force"], check=True)

        # 2. Fixel Correspondence
        subprocess.run(["fixelcorrespondence", subject_fd_file, TEMPLATE_FIXEL_MASK, FD_DIR, f"sub-{sid}.mif", "-force"], check=True)

        # 3. Extract FC
        subprocess.run(["warp2metric", warp_fwd, "-fc", TEMPLATE_FIXEL_MASK, FC_DIR, f"sub-{sid}.mif", "-force"], check=True)

        # 4. Compute FDC
        subprocess.run(["mrcalc", subj_fd_out, subj_fc_out, "-mult", subj_fdc_out, "-force"], check=True)

        # 5. Cleanup
        subprocess.run(["rm", "-rf", tmp_fixel_dir])
        if os.path.exists(subj_fdc_out):
            os.remove(warped_fod)
        
        return f"SUCCESS: {sid}"
    
    except Exception as e:
        return f"FAIL: {sid} - {str(e)}"

def run_level_3():
    df = pd.read_csv(CSV_PATH)
    gender_map = {1: 'M', 0: 'F'}
    subject_list = [(str(int(row['Subject'])), gender_map.get(int(row['Gender']))) for _, row in df.iterrows()]

    print("Executing Super Rescue Protocol (Corrected Syntax)...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(super_rescue, subject_list))
    
    for res in results: print(res)
    subprocess.run(f"rm -rf {METRICS_BASE}/tmp_*", shell=True)


if __name__ == "__main__":
    run_level_1()  # Level 1
    run_level_2()  # Level 2
    run_level_3()  # Level 3