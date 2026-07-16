"""
07_tractography.py

Corresponds to Methods: tractography and connectivity-matrix generation feeding fixel-based statistics. Two levels: the weighted (SIFT2) and unweighted (geometric) fixel connectivity matrices, both required inputs for connectivity-based fixel enhancement (CFE).
"""

import os
import subprocess
import logging
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# ============================================================
# LEVEL 1 -- ACT+SIFT2 tractography, streamline weighting, SIFT2-weighted CFE connectivity matrix (orig. phase7Atckgen.py)
# ============================================================

#!/usr/bin/env python3
"""
phase7Atckgen.py
===========================
FINAL PRODUCTION TRACTOGRAPHY PIPELINE
Nature Neuroscience submission — Agarwal Lab, BRIC-NBRC

Modifications:
- Bypassed all 'file_ok' checks to force complete overwrite/rerun.
- Appended absolute paths to all MRtrix3 commands to enforce the newly built version.
"""


# PATHS
BASE_DIR      = os.environ.get("PROJECT_ROOT", ".")
TEMPLATE_DIR  = os.path.join(BASE_DIR, 'study_template')

# Explicit path to the newly compiled binaries
# Custom-compiled MRtrix3 build directory. Falls back to PATH-resolved
# binaries if not set -- set MRTRIX3_BIN_DIR only if you need a specific build.
NEW_BIN_DIR   = os.environ.get("MRTRIX3_BIN_DIR", "")

FOD_TEMPLATE  = os.path.join(TEMPLATE_DIR, 'wmfod_template.mif')
FIVETT        = os.path.join(TEMPLATE_DIR, '5tt_template.mif')
GMWMI         = os.path.join(TEMPLATE_DIR, 'gmwmi_template.mif')
FIXEL_MASK    = os.path.join(TEMPLATE_DIR, 'template_fixel_mask')

FINAL_TCK     = os.path.join(TEMPLATE_DIR, 'template_10M_ACT_filtered.tck')
SIFT2_WEIGHTS = os.path.join(TEMPLATE_DIR, 'sift2_weights.txt')
SIFT2_MU      = os.path.join(TEMPLATE_DIR, 'sift2_mu.txt')
MATRIX_DIR    = os.path.join(TEMPLATE_DIR, 'matrix_ACT_SIFT2')

os.makedirs(MATRIX_DIR, exist_ok=True)

# HARDWARE CONFIGURATION 
NUM_CHUNKS        = 4          
TRACKS_PER_CHUNK  = 2_500_000  
THREADS_PER_CHUNK = 18         
THREADS_SIFT2     = 70         
THREADS_FIXELCONN = 70
SIFT2_MEM_GB      = 180        

# LOGGING ───────────────
LOG_FILE = os.path.join(TEMPLATE_DIR,
    f"pipeline_ACT_SIFT2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# UTILITIES ─────────────
def run(cmd, step_name):
    """Run a shell command with timing and error capture."""
    log.info(f"START  {step_name}")
    log.info(f"CMD    {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = timedelta(seconds=int(time.time() - t0))

    if result.returncode != 0:
        log.error(f"FAILED {step_name} after {elapsed}")
        log.error(f"STDERR:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"{step_name} failed — see log: {LOG_FILE}")

    log.info(f"DONE   {step_name} [{elapsed}]")
    return result

def streamline_count(tck_path):
    """Return streamline count from tckinfo."""
    result = subprocess.run([os.path.join(NEW_BIN_DIR, 'tckinfo'), tck_path], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if 'count' in line.lower():
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    return int(p)
    return -1


# STEP 0: PRE-FLIGHT CHECKS ─────────────────────────────────────────────────
def preflight():
    log.info("=" * 65)
    log.info("PRE-FLIGHT CHECKS")
    log.info("=" * 65)

    required = {
        'FOD template':   FOD_TEMPLATE,
        '5tt image':      FIVETT,
        'Fixel mask dir': FIXEL_MASK,
        'New Binary Dir': NEW_BIN_DIR # Verifying new build directory
    }
    for label, path in required.items():
        if os.path.exists(path):
            log.info(f"  OK  {label}: {path}")
        else:
            raise FileNotFoundError(f"MISSING {label}: {path}")

    stat = os.statvfs(TEMPLATE_DIR)
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    log.info(f"  Disk free: {free_gb:.1f} GB (need ~25 GB for 10M tractogram)")
    if free_gb < 25:
        log.warning(f"  LOW DISK: {free_gb:.1f} GB free — monitor during run")

def file_ok(path, min_bytes=1024):
    """True if file exists and is non-trivially sized."""
    return os.path.exists(path) and os.path.getsize(path) > min_bytes

# STEP 1: GMWMI ─────────
def make_gmwmi():
    # OVERWRITE: Skipping exists check
    if file_ok(GMWMI):
        log.info(f"GMWMI exists — skipping: {GMWMI}")
        return
    run([
        os.path.join(NEW_BIN_DIR, '5tt2gmwmi'), FIVETT, GMWMI,
        '-force', '-nthreads', '70'
    ], 'GMWMI generation')


# STEP 2: PARALLEL TCKGEN ───────────────────────────────────────────────────
def run_tckgen_chunk(chunk_id):
    chunk_file = os.path.join(TEMPLATE_DIR, f"temp_ACT_chunk_{chunk_id}.tck")

    # OVERWRITE: Skipping exists check
    if file_ok(chunk_file, min_bytes=1_000_000):
       n = streamline_count(chunk_file)
       log.info(f"Chunk {chunk_id}: exists with {n:,} streamlines — skipping")
       return chunk_file

    log.info(f"Chunk {chunk_id}: launching tckgen ({TRACKS_PER_CHUNK:,} tracks)...")
    t0 = time.time()

    cmd = [
        os.path.join(NEW_BIN_DIR, 'tckgen'), FOD_TEMPLATE, chunk_file,
        '-act',        FIVETT,
        '-seed_gmwmi', GMWMI,
        '-select',     str(TRACKS_PER_CHUNK),
        '-cutoff',     '0.06',
        '-minlength',  '10',
        '-maxlength',  '250',
        '-backtrack',
        '-crop_at_gmwmi',
        '-nthreads',   str(THREADS_PER_CHUNK),
        '-force', '-quiet'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = timedelta(seconds=int(time.time() - t0))

    if result.returncode != 0:
        raise RuntimeError(
            f"tckgen chunk {chunk_id} FAILED after {elapsed}:\n{result.stderr[-2000:]}"
        )

    n = streamline_count(chunk_file)
    log.info(f"Chunk {chunk_id}: complete — {n:,} streamlines [{elapsed}]")
    return chunk_file


def run_parallel_tckgen():
    # OVERWRITE: Skipping exists check
    if file_ok(FINAL_TCK, min_bytes=10_000_000):
      n = streamline_count(FINAL_TCK)
      log.info(f"Final tractogram exists with {n:,} streamlines — skipping tckgen")
      return

    log.info("=" * 65)
    log.info(f"TCKGEN: {NUM_CHUNKS} chunks × {TRACKS_PER_CHUNK:,} = "
             f"{NUM_CHUNKS * TRACKS_PER_CHUNK:,} total streamlines")
    log.info(f"        {NUM_CHUNKS} × {THREADS_PER_CHUNK} = "
             f"{NUM_CHUNKS * THREADS_PER_CHUNK} cores utilised")
    log.info("=" * 65)

    chunk_files = []
    with ProcessPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = {executor.submit(run_tckgen_chunk, i): i
                   for i in range(NUM_CHUNKS)}
        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                path = future.result()
                chunk_files.append((chunk_id, path))
            except Exception as e:
                log.error(f"Chunk {chunk_id} raised exception: {e}")
                raise

    chunk_files.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in chunk_files]

    total = 0
    for path in ordered_paths:
        n = streamline_count(path)
        log.info(f"  Chunk {os.path.basename(path)}: {n:,} streamlines")
        total += max(n, 0)
    log.info(f"  Total across chunks: {total:,}")

    log.info("Merging chunks...")
    run([os.path.join(NEW_BIN_DIR, 'tckedit')] + ordered_paths + [FINAL_TCK, '-force', '-quiet'],
        'tckedit merge')

    final_n = streamline_count(FINAL_TCK)
    log.info(f"Final tractogram: {final_n:,} streamlines")

    for path in ordered_paths:
        if os.path.exists(path):
            os.remove(path)
            log.info(f"  Removed: {os.path.basename(path)}")


# STEP 3: SIFT2 ─────────
def run_sift2():
    # OVERWRITE: Skipping exists check
    # if file_ok(SIFT2_WEIGHTS, min_bytes=1_000_000):
    #     log.info(f"SIFT2 weights exist — skipping: {SIFT2_WEIGHTS}")
    #     return

    log.info("=" * 65)
    log.info("SIFT2: Streamline weight estimation")
    log.info(f"       Input:   {FINAL_TCK}")
    log.info(f"       Weights: {SIFT2_WEIGHTS}")
    log.info(f"       Mu:      {SIFT2_MU}")
    log.info("=" * 65)

    run([
        os.path.join(NEW_BIN_DIR, 'tcksift2'),
        FINAL_TCK,
        FOD_TEMPLATE,
        SIFT2_WEIGHTS,
        '-act',      FIVETT,
        '-out_mu',   SIFT2_MU,
        '-nthreads', str(THREADS_SIFT2),
        '-force'
    ], 'tcksift2')

    if os.path.exists(SIFT2_MU):
        with open(SIFT2_MU) as f:
            mu = f.read().strip()
        log.info(f"SIFT2 mu (proportionality coefficient): {mu}")
        log.info("  [METHODS] Report this value in your methods section.")


# STEP 4: FIXELCONNECTIVITY ─────────────────────────────────────────────────
def run_fixelconnectivity():
    # OVERWRITE: Skipping exists check
    # matrix_files = [f for f in os.listdir(MATRIX_DIR)
    #                 if f.endswith('.mif')] if os.path.exists(MATRIX_DIR) else []
    # if len(matrix_files) > 0:
    #     log.info(f"Matrix exists ({len(matrix_files)} files) — skipping fixelconnectivity")
    #     return

    log.info("=" * 65)
    log.info("FIXELCONNECTIVITY: Building SIFT2-weighted CFE matrix")
    log.info(f"  Tractogram: {FINAL_TCK}")
    log.info(f"  Weights:    {SIFT2_WEIGHTS}")
    log.info(f"  Output:     {MATRIX_DIR}")
    log.info("=" * 65)

    run([
        os.path.join(NEW_BIN_DIR, 'fixelconnectivity'),
        FIXEL_MASK,
        FINAL_TCK,
        MATRIX_DIR,
        '-tck_weights_in', SIFT2_WEIGHTS,
        '-nthreads',       str(THREADS_FIXELCONN),
        '-force'
    ], 'fixelconnectivity')

    matrix_files = os.listdir(MATRIX_DIR)
    log.info(f"Matrix directory contains {len(matrix_files)} files")


# STEP 5: FINAL VERIFICATION ────────────────────────────────────────────────
def verify():
    log.info("=" * 65)
    log.info("FINAL VERIFICATION")
    log.info("=" * 65)

    checks = {
        'Tractogram':       FINAL_TCK,
        'SIFT2 weights':    SIFT2_WEIGHTS,
        'SIFT2 mu':         SIFT2_MU,
        'CFE matrix dir':   MATRIX_DIR,
        'GMWMI':            GMWMI,
    }

    all_ok = True
    for label, path in checks.items():
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = 'OK' if exists and size > 1024 else 'MISSING'
        log.info(f"  {status:7}  {label}: {size/1e9:.2f} GB" if size > 1e6
                 else f"  {status:7}  {label}")
        if status == 'MISSING':
            all_ok = False

    final_n = streamline_count(FINAL_TCK)
    log.info(f"\n  Final streamline count: {final_n:,}")

    if os.path.exists(SIFT2_MU):
        with open(SIFT2_MU) as f:
            mu = f.read().strip()
        log.info(f"  SIFT2 mu:               {mu}")

    matrix_n = len(os.listdir(MATRIX_DIR)) if os.path.exists(MATRIX_DIR) else 0
    log.info(f"  CFE matrix files:       {matrix_n}")

    log.info("")
    if all_ok:
        log.info("  ALL CHECKS PASSED")
        log.info("")
        log.info("  Next step: update phase7B_unified_filtered_stats.py")
        log.info(f"    MATRIX_DIR    = '{MATRIX_DIR}'")
        log.info(f"    NEW_STATS_DIR = '.../stats/newcfestats_ACT_SIFT2'")
        log.info(f"  Log file: {LOG_FILE}")
    else:
        log.error("  ONE OR MORE CHECKS FAILED — review log before proceeding")


# ENTRY POINT ───────────

def run_level_1():
    t_start = time.time()
    log.info("=" * 65)
    log.info("PHASE 7A — FINAL ACT + SIFT2 TRACTOGRAPHY PIPELINE (OVERWRITE MODE)")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Log:     {LOG_FILE}")
    log.info("=" * 65)

    try:
        preflight()
        make_gmwmi()
        run_parallel_tckgen()
        run_sift2()
        run_fixelconnectivity()
        verify()
    except Exception as e:
        log.error(f"PIPELINE FAILED: {e}")
        raise
    finally:
        elapsed = timedelta(seconds=int(time.time() - t_start))
        log.info(f"\nTotal wall time: {elapsed}")

# ============================================================
# LEVEL 2 -- unweighted (geometric) fixel connectivity matrix, also required for CFE (orig. phase7A2tckgenunweighted.py)
# ============================================================

# --- CONFIGURATION ---
# Paths to your specific binaries and data
FIXEL_CONN_BIN = os.path.join(os.environ.get("MRTRIX3_BIN_DIR", ""), "fixelconnectivity") \
    if os.environ.get("MRTRIX3_BIN_DIR") else "fixelconnectivity"
TEMPLATE_MASK = os.path.join(os.environ.get("PROJECT_ROOT", "."), "study_template/template_fixel_mask")
TRACTOGRAM = os.path.join(os.environ.get("PROJECT_ROOT", "."), "study_template/template_10M_ACT_filtered.tck")
OUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "study_template/matrix_ACT_unweighted")

# System Resource Management
THREADS = "70"

def run_unweighted_connectivity():
    """
    Executes the geometric (unweighted) fixel connectivity mapping.
    This is required for the correct CFE enhancement of the FC metric.
    """
    
    # Ensure output directory exists or MRtrix will handle it with -force
    if not os.path.exists(OUT_DIR):
        print(f"[{datetime.now()}] Creating output directory: {OUT_DIR}")
        os.makedirs(OUT_DIR, exist_ok=True)

    # Build the command list
    cmd = [
        FIXEL_CONN_BIN,
        TEMPLATE_MASK,
        TRACTOGRAM,
        OUT_DIR,
        "-nthreads", THREADS,
        "-force"
    ]

    start_time = time.time()
    print(f"[{datetime.now()}] STARTING: fixelconnectivity (Unweighted Matrix)")
    print(f"Mapping 10M streamlines to fixels using {THREADS} threads...")
    print("-" * 60)

    try:
        # Run the command and capture output
        # Using subprocess.run with check=True will raise an error if the binary fails
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Log completion details
        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        print(f"[{datetime.now()}] SUCCESS: Unweighted matrix generated in {duration:.2f} minutes.")
        print(f"Output saved to: {OUT_DIR}")

    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] CRITICAL ERROR: fixelconnectivity failed.")
        print(f"Return Code: {e.returncode}")
        print(f"Error Message:\n{e.stderr}")

def run_level_2():
    run_unweighted_connectivity()


if __name__ == "__main__":
    run_level_1()  # Level 1
    run_level_2()  # Level 2