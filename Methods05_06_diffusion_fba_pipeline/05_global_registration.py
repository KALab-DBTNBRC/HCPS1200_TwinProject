import os
import pandas as pd
import subprocess
import logging
import datetime

# CONFIGURATION
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
# NOTE: this derived table carries SSAGA/behavioral fields sourced from
# HCP Restricted-Access data. Never committed to this repository.
CSV_PATH = os.environ.get(
    "RESTRICTED_DERIVED_TWIN_TABLE",
    os.path.join(BASE_DIR, 'twintables/Twins240_DTI - Sheet1.csv')
)
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')
FOD_TEMPLATE = os.path.join(TEMPLATE_DIR, 'wmfod_template.mif')

# Inputs
FOD_INPUT_DIR = os.path.join(BASE_DIR, 'processed/template_inputs/fods')
MASK_INPUT_DIR = os.path.join(BASE_DIR, 'processed/template_inputs/masks')

# Output for the 238 Subject Warps
REG_OUTPUT_DIR = os.path.join(BASE_DIR, 'registration')
os.makedirs(REG_OUTPUT_DIR, exist_ok=True)

# LOG SETUP
LOG_FILE = os.path.join(REG_OUTPUT_DIR, f"global_registration_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger()

def run_global_registration():
    logger.info("--- Starting Phase 4: Global Registration (238 Subjects) ---")
    
    df = pd.read_csv(CSV_PATH)
    gender_map = {1: 'M', 0: 'F'}
    
    total = len(df)
    success = 0

    for idx, row in df.iterrows():
        sid = str(int(row['Subject']))
        sex = gender_map.get(int(row['Gender']), 'Unknown')
        filename = f"sub-{sid}_{sex}.mif"
        
        subj_fod = os.path.join(FOD_INPUT_DIR, filename)
        subj_mask = os.path.join(MASK_INPUT_DIR, filename)
        
        # Warp outputs: Subject-to-Template
        warp_fwd = os.path.join(REG_OUTPUT_DIR, f"sub-{sid}_warp_fwd.mif")
        warp_inv = os.path.join(REG_OUTPUT_DIR, f"sub-{sid}_warp_inv.mif")
        
        if os.path.exists(subj_fod):
            if os.path.exists(warp_fwd):
                logger.info(f"[{idx+1}/{total}] Skipping {sid}: Warp already exists.")
                success += 1
                continue
            
            logger.info(f"[{idx+1}/{total}] Registering {sid} to study template...")
            
            # mrregister
            cmd = [
                "mrregister", subj_fod, FOD_TEMPLATE,
                "-mask1", subj_mask,
                "-nl_warp", warp_fwd, warp_inv,
                "-force", "-nthreads", "40"
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                success += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"Registration failed for {sid}: {e.stderr.decode()}")
        else:
            logger.warning(f"FOD file not found for {sid}: {subj_fod}")

    logger.info(f"PHASE 4 COMPLETE. Registered {success}/{total} subjects.")

if __name__ == "__main__":
    run_global_registration()