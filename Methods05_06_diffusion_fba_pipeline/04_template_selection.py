import os
import pandas as pd
import subprocess
import logging
import datetime
import shutil

# CONFIGURATION
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
# NOTE: this derived table carries SSAGA/behavioral fields sourced from
# HCP Restricted-Access data. Never committed to this repository.
CSV_PATH = os.environ.get(
    "RESTRICTED_DERIVED_TWIN_TABLE",
    os.path.join(BASE_DIR, 'twintables/Twins240_DTI - Sheet1.csv')
)
FOD_INPUT_DIR = os.path.join(BASE_DIR, 'processed/template_inputs/fods')
MASK_INPUT_DIR = os.path.join(BASE_DIR, 'processed/template_inputs/masks')
OUTPUT_DIR = os.path.join(BASE_DIR, 'study_template')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOG SETUP
LOG_FILE = os.path.join(OUTPUT_DIR, f"template_build_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger()

def run_phase_3_final():
    logger.info("--- Phase 3: Study-Specific Population Template (0/1 to F/M Mapping) ---")
    
    # LOAD DATA
    df = pd.read_csv(CSV_PATH)
    
    # APPLY SSAGA CRITERIA (DSM4 Abuse/Dependence: 5=Yes) [cite: 736, 738]
    df['is_AUD'] = ((df['SSAGA_Alc_D4_Ab_Dx'] == 5) | (df['SSAGA_Alc_D4_Dp_Dx'] == 5)).astype(int)
    
    # Smoking Status [cite: 782, 783]
    if 'SSAGA_TB_Still_Smoking' in df.columns:
        df['is_Smoker'] = ((df['SSAGA_TB_Still_Smoking'] == 1) | (df['SSAGA_TB_Smoking_History'] == 3)).astype(int)
    else:
        df['is_Smoker'] = 0

    # SELECT INDEPENDENT SUBSET (1 Twin per family)
    # Using TwinPairID ensures we don't pick two biological twins for the same template
    independent_df = df.groupby('TwinPairID').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
    
    # Select 40 subjects for the template build [cite: 621]
    template_subset = independent_df.sample(n=min(40, len(independent_df)), random_state=1)
    logger.info(f"Selected {len(template_subset)} independent subjects for the template build.")

    # PREPARE INPUT DIRECTORIES
    subset_fods = os.path.join(OUTPUT_DIR, 'subset_fods')
    subset_masks = os.path.join(OUTPUT_DIR, 'subset_masks')
    
    for d in [subset_fods, subset_masks]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    # MAPPING DICTIONARY & SYMLINKING
    # M=1, F=0
    gender_map = {1: 'M', 0: 'F'}
    
    success_count = 0
    for _, row in template_subset.iterrows():
        sid = str(int(row['Subject']))
        gender_val = int(row['Gender'])
        sex_char = gender_map.get(gender_val, 'Unknown')
        
        filename = f"sub-{sid}_{sex_char}.mif"
        
        src_fod = os.path.join(FOD_INPUT_DIR, filename)
        src_mask = os.path.join(MASK_INPUT_DIR, filename)
        
        if os.path.exists(src_fod) and os.path.exists(src_mask):
            os.symlink(src_fod, os.path.join(subset_fods, filename))
            os.symlink(src_mask, os.path.join(subset_masks, filename))
            success_count += 1
        else:
            logger.error(f"File NOT FOUND for Subject {sid} (Gender {gender_val}->{sex_char}): {src_fod}")

    if success_count < 2:
        logger.error(f"Critical failure: Only {success_count} files found. Check your folder paths.")
        return

    # RUN population_template [cite: 79, 289]
    template_mif = os.path.join(OUTPUT_DIR, 'wmfod_template.mif')
    cmd = [
        "population_template", subset_fods,
        "-mask_dir", subset_masks,
        template_mif,
        "-type", "rigid_affine_nonlinear", # Standard diffeomorphic iterative approach [cite: 162]
        "-voxel_size", "1.25",              # Native HCP resolution [cite: 394, 439]
        "-nthreads", "40"
    ]

    logger.info(f"Initiating MRtrix3 build with {success_count} subjects...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logger.info(f"[MRtrix3] {line.strip()}")
    process.wait()

    if process.returncode == 0:
        logger.info("PHASE 3 SUCCESS: Study-specific FOD template built.")
    else:
        logger.error("Phase 3 failed in MRtrix3. Check logs.")

if __name__ == "__main__":
    run_phase_3_final()