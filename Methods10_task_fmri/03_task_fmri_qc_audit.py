import os
import pandas as pd
import numpy as np
import logging
import datetime
import hashlib

# --- Configuration ---
SCRIPT_VERSION = "1.0.0"
BASE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "Organized")
OUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "tfMRITables")
OUT_FILE = os.path.join(OUT_DIR, "QC_Audit_Report.csv")
LOG_FILE = os.path.join(os.path.join(os.environ.get("PROJECT_ROOT", "."), "logs"), f"Audit_0AB_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # Keeps output visible in the terminal
    ]
)

# Exact EV requirements per task
EV_REQUIREMENTS = {
    "GAMBLING": ['win.txt', 'loss.txt', 'win_event.txt', 'loss_event.txt', 'neut_event.txt', 'Sync.txt'],
    "EMOTION": ['fear.txt', 'neut.txt', 'Sync.txt'],
    "WM": ['2bk_faces.txt', '2bk_body.txt', '2bk_places.txt', '2bk_tools.txt', '0bk_faces.txt', '0bk_body.txt', '0bk_places.txt', '0bk_tools.txt', 'all_bk_cor.txt', 'all_bk_err.txt', 'Sync.txt'],
    "SOCIAL": ['mental.txt', 'rnd.txt', 'Sync.txt'],
    "RELATIONAL": ['relation.txt', 'match.txt', 'Sync.txt'],
    "LANGUAGE": ['story.txt', 'math.txt', 'Sync.txt'],
    "MOTOR": ['lf.txt', 'lh.txt', 'rf.txt', 'rh.txt', 't.txt', 'Sync.txt']
}

def generate_file_hash(filepath):
    """Generates an MD5 hash of the output file for provenance."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# --- Execution ---
logging.info(f"--- STARTING PHASE 0A & 0B AUDIT ---")
logging.info(f"Script Version: {SCRIPT_VERSION}")
logging.info(f"Base Directory: {BASE_DIR}")

TASKS = list(EV_REQUIREMENTS.keys())
RUNS = ["LR", "RL"]
results = []

for task in TASKS:
    task_dir = os.path.join(BASE_DIR, task)
    if not os.path.exists(task_dir):
        logging.warning(f"Task directory missing: {task_dir}")
        continue
        
    subjects = [d for d in os.listdir(task_dir) if d.isdigit() and len(d) == 6]
    logging.info(f"Scanning {task}: Found {len(subjects)} potential subjects.")
    
    for sub in subjects:
        for run in RUNS:
            run_dir = os.path.join(task_dir, sub, run)
            if not os.path.exists(run_dir):
                continue
            
            row = {
                'Subject': sub, 'Task': task, 'Run': run,
                'CIFTI_Present': False, 'Missing_EVs': '', 'Sparse_EVs': '',
                'Mean_Abs_RMS': np.nan, 'Max_Rel_RMS': np.nan, 'Pct_Scrubbed': np.nan,
                'EXCLUDE_RUN': False, 'Exclusion_Reason': ''
            }
            
            # 1. CIFTI Check
            cifti_file = os.path.join(run_dir, f"tfMRI_{task}_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii")
            if os.path.exists(cifti_file):
                row['CIFTI_Present'] = True
            else:
                row['EXCLUDE_RUN'] = True
                row['Exclusion_Reason'] += "Missing CIFTI; "

            # 2. EV Audit (Phase 0A)
            missing_evs = []
            sparse_evs = []
            ev_dir = os.path.join(run_dir, "EVs")
            
            for req_ev in EV_REQUIREMENTS[task]:
                ev_path = os.path.join(ev_dir, req_ev)
                if not os.path.exists(ev_path):
                    missing_evs.append(req_ev)
                elif req_ev != 'Sync.txt':
                    try:
                        ev_data = pd.read_csv(ev_path, sep='\t', header=None)
                        if len(ev_data) < 3:
                            sparse_evs.append(f"{req_ev}({len(ev_data)})")
                    except pd.errors.EmptyDataError:
                        sparse_evs.append(f"{req_ev}(0)")
            
            if missing_evs:
                row['Missing_EVs'] = "|".join(missing_evs)
                row['EXCLUDE_RUN'] = True
                row['Exclusion_Reason'] += "Missing EVs; "
            if sparse_evs:
                row['Sparse_EVs'] = "|".join(sparse_evs)

            # 3. Motion QC (Phase 0B)
            abs_rms_file = os.path.join(run_dir, "Movement_AbsoluteRMS.txt")
            rel_rms_file = os.path.join(run_dir, "Movement_RelativeRMS.txt")
            
            if os.path.exists(abs_rms_file):
                abs_data = np.loadtxt(abs_rms_file)
                mean_abs = np.mean(abs_data)
                row['Mean_Abs_RMS'] = round(mean_abs, 4)
                if mean_abs > 0.3:
                    row['EXCLUDE_RUN'] = True
                    row['Exclusion_Reason'] += f"High Abs RMS ({mean_abs:.2f}); "
            
            if os.path.exists(rel_rms_file):
                rel_data = np.loadtxt(rel_rms_file)
                max_rel = np.max(rel_data)
                pct_scrubbed = np.mean(rel_data > 1.5) * 100
                row['Max_Rel_RMS'] = round(max_rel, 4)
                row['Pct_Scrubbed'] = round(pct_scrubbed, 2)
                
                if pct_scrubbed > 20.0:
                    row['EXCLUDE_RUN'] = True
                    row['Exclusion_Reason'] += f"High Scrubbing ({pct_scrubbed:.1f}%); "
            
            results.append(row)

# --- Report Generation & Hash ---
df = pd.DataFrame(results)
df.to_csv(OUT_FILE, index=False)

output_hash = generate_file_hash(OUT_FILE)

logging.info("--- AUDIT COMPLETE ---")
logging.info(f"Total Runs Analyzed: {len(df)}")
logging.info(f"Total Runs Flagged for Exclusion: {df['EXCLUDE_RUN'].sum()}")
logging.info(f"Report File: {OUT_FILE}")
logging.info(f"Output Hash (MD5): {output_hash}")

# Calculate exclusions per task
logging.info("Exclusions by Task:")
exclusion_counts = df[df['EXCLUDE_RUN'] == True]['Task'].value_counts()
if not exclusion_counts.empty:
    for task, count in exclusion_counts.items():
        logging.info(f"  - {task}: {count} runs")
else:
    logging.info("  - None!")