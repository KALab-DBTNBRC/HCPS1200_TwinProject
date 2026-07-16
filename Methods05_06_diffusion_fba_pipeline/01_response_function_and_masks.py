"""
01_response_function_and_masks.py

Corresponds to Methods: "Diffusion MRI -- fixel-based analysis (FBA)",
opening steps -- bias correction and Dhollander response function
estimation. Two levels, fused because Level 2 is a repair/completion pass
over the exact output of Level 1, not an independent step.
"""

import os
import subprocess
import logging
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# CONFIGURATION
RAW_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "restructured_raw")
OUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "processed")
MAX_WORKERS = 10

os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, f"phase1_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])


def run_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED: {' '.join(cmd_list)}\nSTDERR: {e.stderr.strip()}")
        return False



# LEVEL 1 -- per-subject bias correction and Dhollander response
# function estimation (orig. phase1Aantsdhollander.py)


def process_subject(sub_info):
    zygosity, pair, sub_folder, sub_in = sub_info
    sub_out = os.path.join(OUT_DIR, zygosity, pair, sub_folder)
    os.makedirs(sub_out, exist_ok=True)

    dwi_nii = glob.glob(os.path.join(sub_in, "*_dwi.nii.gz"))[0]
    bval = glob.glob(os.path.join(sub_in, "*_dwi.bval"))[0]
    bvec = glob.glob(os.path.join(sub_in, "*_dwi.bvec"))[0]
    mask_nii = glob.glob(os.path.join(sub_in, "*_mask.nii.gz"))[0]

    dwi_mif = os.path.join(sub_out, "dwi_raw.mif")
    mask_mif = os.path.join(sub_out, "mask.mif")
    if not os.path.exists(dwi_mif):
        run_cmd(["mrconvert", dwi_nii, dwi_mif, "-fslgrad", bvec, bval, "-datatype", "float32", "-force"])
        run_cmd(["mrconvert", mask_nii, mask_mif, "-datatype", "bit", "-force"])

    dwi_bias = os.path.join(sub_out, "dwi_biascorr.mif")
    if not os.path.exists(dwi_bias):
        run_cmd(["dwibiascorrect", "ants", dwi_mif, dwi_bias, "-mask", mask_mif, "-force"])

    wm_txt = os.path.join(sub_out, "res_wm.txt")
    gm_txt = os.path.join(sub_out, "res_gm.txt")
    csf_txt = os.path.join(sub_out, "res_csf.txt")
    if not os.path.exists(wm_txt):
        run_cmd(["dwi2response", "dhollander", dwi_bias, wm_txt, gm_txt, csf_txt, "-mask", mask_mif, "-force"])

    return f"Finished: {sub_folder}"


def run_phase_1_parallel():
    subjects_to_process = []
    for zyg in ["MZ", "DZ"]:
        zyg_path = os.path.join(RAW_DIR, zyg)
        for pair in os.listdir(zyg_path):
            pair_path = os.path.join(zyg_path, pair)
            for sub in os.listdir(pair_path):
                sub_path = os.path.join(pair_path, sub)
                if os.path.isdir(sub_path):
                    subjects_to_process.append((zyg, pair, sub, sub_path))

    logging.info(f"Starting parallel processing with {MAX_WORKERS} workers for {len(subjects_to_process)} subjects.")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_subject, subjects_to_process))
    logging.info("=== All subjects processed in Level 1 ===")



# LEVEL 2 -- repair pass: regenerate response/voxel files for any
# subject where Level 1's output is incomplete (orig. phase1Bvoxelgenerator.py)


def repair_voxels(sub_path):
    dwi_bias = os.path.join(sub_path, "dwi_biascorr.mif")
    mask_mif = os.path.join(sub_path, "mask.mif")
    wm_txt = os.path.join(sub_path, "res_wm.txt")
    gm_txt = os.path.join(sub_path, "res_gm.txt")
    csf_txt = os.path.join(sub_path, "res_csf.txt")
    voxels_mif = os.path.join(sub_path, "voxels_dhollander.mif")

    if os.path.exists(dwi_bias) and not os.path.exists(voxels_mif):
        run_cmd(["dwi2response", "dhollander", dwi_bias, wm_txt, gm_txt, csf_txt,
                 "-mask", mask_mif, "-voxels", voxels_mif, "-force"])
        return f"Repaired: {os.path.basename(sub_path)}"
    return None


def run_phase_1_repair():
    subject_dirs = glob.glob(os.path.join(OUT_DIR, "*", "*", "sub-*"))
    logging.info(f"Checking {len(subject_dirs)} subjects for missing voxel files...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(repair_voxels, subject_dirs))
    repaired = [r for r in results if r is not None]
    logging.info(f"Repair complete. Generated {len(repaired)} voxel maps.")


if __name__ == "__main__":
    run_phase_1_parallel()
    run_phase_1_repair()
