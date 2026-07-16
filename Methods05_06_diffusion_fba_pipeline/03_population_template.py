import os
import subprocess
import logging
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# CONFIGURATION
BASE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "processed")
GROUP_RES_DIR = os.path.join(BASE_DIR, "group_responses")
TEMPLATE_INPUT_DIR = os.path.join(BASE_DIR, "template_inputs")
TEMPLATE_OUT_DIR = os.path.join(BASE_DIR, "population_template")
MAX_WORKERS = 6

os.makedirs(TEMPLATE_INPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(TEMPLATE_INPUT_DIR, "fods"), exist_ok=True)
os.makedirs(os.path.join(TEMPLATE_INPUT_DIR, "masks"), exist_ok=True)

# LOG
LOG_FILE = os.path.join(BASE_DIR, f"weekend_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

def run_cmd(cmd_list):
    subprocess.run(cmd_list, check=True, capture_output=True, text=True)

def process_fod(sub_path):
    sub_id = os.path.basename(sub_path)
    dwi_bias = os.path.join(sub_path, "dwi_biascorr.mif")
    mask = os.path.join(sub_path, "mask.mif")
    
    # Master Responses
    avg_wm = os.path.join(GROUP_RES_DIR, "master_wm.txt")
    avg_gm = os.path.join(GROUP_RES_DIR, "master_gm.txt")
    avg_csf = os.path.join(GROUP_RES_DIR, "master_csf.txt")

    # Outputs
    wm_fod = os.path.join(sub_path, "wmfod.mif")
    gm_fod = os.path.join(sub_path, "gmfod.mif")
    csf_fod = os.path.join(sub_path, "csffod.mif")
    
    wm_norm = os.path.join(sub_path, "wmfod_norm.mif")
    gm_norm = os.path.join(sub_path, "gmfod_norm.mif")
    csf_norm = os.path.join(sub_path, "csffod_norm.mif")

    try:
        if not os.path.exists(wm_fod):
            # SS3T-CSD
            run_cmd(["dwi2fod", "msmt_csd", dwi_bias, avg_wm, wm_fod, avg_gm, gm_fod, avg_csf, csf_fod, "-mask", mask, "-force"])
        
        if not os.path.exists(wm_norm):
            # Normalise
            run_cmd(["mtnormalise", wm_fod, wm_norm, gm_fod, gm_norm, csf_fod, csf_norm, "-mask", mask, "-force"])

        # 3. Symlink to template input directory for easy population_template execution
        target_fod = os.path.join(TEMPLATE_INPUT_DIR, "fods", f"{sub_id}.mif")
        target_mask = os.path.join(TEMPLATE_INPUT_DIR, "masks", f"{sub_id}.mif")
        
        if not os.path.exists(target_fod): os.symlink(wm_norm, target_fod)
        if not os.path.exists(target_mask): os.symlink(mask, target_mask)

        return f"Completed FOD & Norm for: {sub_id}"
    except Exception as e:
        return f"ERROR on {sub_id}: {str(e)}"

def main():
    logging.info("=== STARTING FOD GENERATION ===")
    subject_dirs = glob.glob(os.path.join(BASE_DIR, "*", "*", "sub-*"))
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_fod, subject_dirs):
            logging.info(result)

    logging.info("=== FODs COMPLETE. LAUNCHING POPULATION TEMPLATE ===")
    
    # The Population Template
    template_cmd = [
        "population_template", 
        os.path.join(TEMPLATE_INPUT_DIR, "fods"), 
        "-mask_dir", os.path.join(TEMPLATE_INPUT_DIR, "masks"), 
        TEMPLATE_OUT_DIR, 
        "-voxel_size", "1.25", 
        "-type", "rigid_affine_nonlinear", 
        "-linear_transformations_dir", os.path.join(TEMPLATE_OUT_DIR, "linear_transforms"),
        "-nonlinear_transformations_dir", os.path.join(TEMPLATE_OUT_DIR, "nonlinear_transforms"),
        "-force"
    ]
    
    logging.info(f"Executing: {' '.join(template_cmd)}")

process = subprocess.Popen(template_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    logging.info(f"[MRtrix3] {line.strip()}")
process.wait()

if process.returncode != 0:
    logging.error("population_template FAILED")
else:
    logging.info("population_template completed successfully.")

if __name__ == "__main__":
    main()