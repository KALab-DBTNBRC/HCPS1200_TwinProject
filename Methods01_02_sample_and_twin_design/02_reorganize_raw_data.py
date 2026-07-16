import os
import shutil
import csv
import logging
from pathlib import Path

SOURCE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "raw")
TARGET_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "restructured_raw")
# NOTE: This file contains HCP Restricted-Access data (zygosity, family
# structure). It is never included in this repository. Obtain your own copy
# under own HCP Restricted Data Use Agreement and set the path below,
# or override via the RESTRICTED_TWIN_TABLE environment variable.
MASTER_CSV = os.environ.get(
    "RESTRICTED_TWIN_TABLE",
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/Twins240_DTI - Sheet1.csv")
)

EXCLUDE_PAIRS = ["Pair41"]

GENDER_MAP = {"0": "F", "1": "M"}

FILE_MAPPING = {
    "dwi.nii.gz": "dwi.nii.gz",
    "bvals": "dwi.bval",
    "bvecs": "dwi.bvec",
    "mask.nii.gz": "mask.nii.gz",
    "T1.nii.gz": "T1w.nii.gz"
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def restructure_data():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    with open(MASTER_CSV, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            sub_id = row['Subject']
            pair_id = row['TwinPairID']
            zygosity = row['ZygosityGT1']
            gender_raw = row['Gender']
            sex = GENDER_MAP.get(gender_raw, "U") # U for Unknown if not 0 or 1

            if pair_id in EXCLUDE_PAIRS:
                logging.warning(f"Skipping {sub_id}: Part of excluded {pair_id}")
                continue

            # [Zygosity] / [PairID] / sub-[ID]_[Sex]
            sub_folder_name = f"sub-{sub_id}_{sex}"
            new_sub_path = os.path.join(TARGET_DIR, zygosity, pair_id, sub_folder_name)
            old_sub_path = os.path.join(SOURCE_DIR, sub_id)

            if not os.path.exists(old_sub_path):
                logging.error(f"Source folder not found for {sub_id} at {old_sub_path}")
                continue

            os.makedirs(new_sub_path, exist_ok=True)

            for old_name, new_suffix in FILE_MAPPING.items():
                src_file = os.path.join(old_sub_path, old_name)
                
                if os.path.exists(src_file):
                    new_filename = f"{sub_folder_name}_{zygosity}_{new_suffix}"
                    dst_file = os.path.join(new_sub_path, new_filename)
                    
                    shutil.copy2(src_file, dst_file)
                else:
                    logging.warning(f"Missing file {old_name} for subject {sub_id}")

            logging.info(f"Successfully organized: {sub_id} ({zygosity} {pair_id})")

    print(f"\nRestructuring Complete. New data located at: {TARGET_DIR}")

if __name__ == "__main__":
    restructure_data()