"""
09_binarise_and_network_masks.py

Corresponds to Methods: network mask construction in fixel space, prerequisite to per-network metric extraction. Two levels: binarisation, then per-network mask construction from the binarised template.
"""

import subprocess
import os
import shutil
import logging
from datetime import datetime



# LEVEL 1 -- binarise template fixel masks (orig. phase8binarisefixelmasks.py)


# --- CONFIGURATION ---
BASE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_masks/fixel")
NETWORKS = ['Reward', 'Salience', 'DMN', 'Olfactory']

def binarize_fixels():
    print("--- Binarizing Fixel Masks ---")
    
    for net in NETWORKS:
        # Construct the paths based on your peer's naming convention
        input_mask = os.path.join(BASE_DIR, f'{net}_fixel_mask.mif')
        output_mask = os.path.join(BASE_DIR, f'{net}_fixel_mask_bin.mif')
        
        if os.path.exists(input_mask):
            print(f"Processing {net}...")
            # Thresholding at 0.5 ensures that any fixel with data becomes 1, 
            # and everything else becomes 0.
            cmd = ['mrthreshold', input_mask, '-abs', '0.5', output_mask, '-force']
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Verification: Count the resulting fixels
                stats = subprocess.check_output(['mrstats', output_mask, '-output', 'count', '-ignorezero']).decode().strip()
                print(f"  > Success. Final {net} count: {stats} fixels.")
            except subprocess.CalledProcessError as e:
                print(f"  > Error processing {net}: {e.stderr}")
        else:
            print(f"  > Skipping {net}: File not found at {input_mask}")

def run_level_1():
    binarize_fixels()


# LEVEL 2 -- construct per-network fixel masks from binarised template (orig. phase8networkmaskmaker.py)


"""
phase8_final_maskmaker.py
=========================
Generates clean, final network masks for all downstream analyses.

Output structure:
    final_masks/
        voxel/
            Reward_voxel_mask.mif        ← 3D binary, for DTI FA/MD/RD/AD extraction
            Salience_voxel_mask.mif
            DMN_voxel_mask.mif
            Olfactory_voxel_mask.mif
        fixel/
            directions.mif               ← copied from template fixel mask
            index.mif                    ← copied from template fixel mask
            Reward_fixel_mask.mif        ← 1D fixel-space, for FBA mrstats
            Salience_fixel_mask.mif
            DMN_fixel_mask.mif
            Olfactory_fixel_mask.mif

All Glasser indices confirmed from HCP-Multi-Modal-Parcellation-1_0.xml
All Tian indices confirmed from Tian_Subcortex_S3_3T_label.txt
"""


# PATHS 
BASE_DIR     = os.environ.get("PROJECT_ROOT", ".")
REF_DIR      = os.path.join(BASE_DIR, 'reference_atlas')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')

GLASSER = os.path.join(REF_DIR, 'Glasser_in_Template_FINAL.nii.gz')
TIAN    = os.path.join(REF_DIR, 'Tian_in_Template_FINAL.nii.gz')
JHU     = os.path.join(REF_DIR, 'JHU_in_Template_FINAL.nii.gz')

# Template fixel mask — source for index.mif and directions.mif
TEMPLATE_FIXEL_MASK = os.path.join(TEMPLATE_DIR, 'template_fixel_mask')

# FINAL OUTPUT DIRECTORIES 
FINAL_DIR       = os.path.join(BASE_DIR, 'final_masks')
VOXEL_DIR       = os.path.join(FINAL_DIR, 'voxel')
FIXEL_DIR       = os.path.join(FINAL_DIR, 'fixel')
TMP_DIR         = os.path.join(FINAL_DIR, 'tmp')

for d in [VOXEL_DIR, FIXEL_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)

# LOGGING 
LOG_FILE = os.path.join(FINAL_DIR,
    f"maskmaker_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger()


# VERIFIED NETWORK DEFINITIONS 
# Coordinates identical to phase8networkmaskmaker.py

NETWORKS = {

    'Reward': {
        GLASSER: [
            90, 1090,  72, 1072,  88, 1088,  65, 1065,   # vmPFC / mPFC
            91, 1091,  92, 1092,  93, 1093, 166, 1166,   # OFC
            164, 1164, 165, 1165,                          # Subgenual ACC
            111, 1111, 112, 1112,                          # Anterior insula
            61, 1061,  180, 1180,                          # ACC
        ],
        TIAN: [
            22, 23, 47, 48,                                # NAcc bilateral
            15, 16, 17, 18, 40, 41, 42, 43,               # Caudate bilateral
            11, 12, 13, 14, 36, 37, 38, 39,               # Putamen bilateral
            19, 20, 44, 45,                                # Amygdala bilateral
        ],
    },

    'Salience': {
        GLASSER: [
            111, 1111, 112, 1112, 109, 1109,              # Anterior insula core
            57, 1057,  59, 1059,  40, 1040,               # dACC
            79, 1079,  80, 1080,  81, 1081,               # IFJ / IFS
            113, 1113, 115, 1115, 114, 1114,              # Frontal operculum
        ],
        TIAN: [
            19, 20, 44, 45,                                # Amygdala bilateral
            9, 34,                                         # Mediodorsal thalamus
        ],
    },

    'DMN': {
        GLASSER: [
            161, 1161, 162, 1162,  35, 1035,              # Posterior cingulate
            33, 1033,  34, 1034,   14, 1014,              # Cingulate cont.
            65, 1065,  72, 1072,   88, 1088,              # mPFC
            27, 1027,  30, 1030,                           # Precuneus
            150, 1150, 143, 1143,                          # Angular gyrus
            128, 1128, 129, 1129,                          # Posterior temporal
            126, 1126, 155, 1155,  127, 1127,             # Parahippocampal
        ],
        TIAN: [
            1, 2, 3, 4, 26, 27, 28, 29,                   # Hippocampus bilateral
        ],
    },

    'Olfactory': {
        GLASSER: [
            110, 1110,                                     # Piriform cortex
            118, 1118, 119, 1119,                          # Entorhinal / perirhinal
            131, 1131, 172, 1172,                          # Temporal pole
            93, 1093,  166, 1166,  92, 1092,              # Olfactory OFC
            164, 1164,                                     # Subcallosal
        ],
        TIAN: [
            19, 20, 44, 45,                                # Amygdala bilateral
            1, 2, 3, 4, 26, 27, 28, 29,                   # Hippocampus bilateral
        ],
        JHU: [
            16, 17,                                        # Uncinate fasciculus L/R
        ],
    },
}


# FUNCTIONS
def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n  {result.stderr[:300]}")


def make_atlas_mask(atlas, label_ids, out_path):
    """Binary voxel mask for a list of label IDs from one atlas."""
    cmd = [atlas, str(label_ids[0]), '-eq']
    for lid in label_ids[1:]:
        cmd += [atlas, str(lid), '-eq', '-add']
    cmd += ['0', '-gt', out_path, '-force']
    run(['mrcalc'] + cmd)


def build_voxel_mask(net_name, region_map):
    """Combine atlas-specific masks into one binary 3D voxel mask."""
    log.info(f"  Building voxel mask: {net_name}")
    tmp_files = []

    for atlas, ids in region_map.items():
        atlas_label = os.path.basename(atlas).split('_')[0]
        tmp = os.path.join(TMP_DIR, f'{net_name}_{atlas_label}.mif')
        log.info(f"    {atlas_label}: {len(ids)} region IDs")
        make_atlas_mask(atlas, ids, tmp)
        tmp_files.append(tmp)

    out_path = os.path.join(VOXEL_DIR, f'{net_name}_voxel_mask.mif')

    if len(tmp_files) == 1:
        run(['mrconvert', tmp_files[0], out_path, '-force'])
    else:
        cmd = [tmp_files[0]]
        for t in tmp_files[1:]:
            cmd += [t, '-add']
        cmd += ['0', '-gt', out_path, '-force']
        run(['mrcalc'] + cmd)

    # Cleanup tmp files for this network
    for t in tmp_files:
        if os.path.exists(t):
            os.remove(t)

    # Verify output
    result = subprocess.run(
        ['mrstats', out_path, '-output', 'mean', '-ignorezero'],
        capture_output=True, text=True)
    log.info(f"    → {out_path}")
    return out_path


def build_fixel_mask(net_name, voxel_mask_path):
    """Convert voxel mask to fixel space using voxel2fixel."""
    log.info(f"  Building fixel mask: {net_name}")
    out_path = os.path.join(FIXEL_DIR, f'{net_name}_fixel_mask.mif')

    run([
        'voxel2fixel',
        voxel_mask_path,
        TEMPLATE_FIXEL_MASK,
        FIXEL_DIR,
        f'{net_name}_fixel_mask.mif',
        '-force'
    ])

    # Count non-zero fixels
    result = subprocess.run(
        ['mrstats', out_path, '-output', 'count', '-ignorezero'],
        capture_output=True, text=True)
    n_fixels = result.stdout.strip()
    log.info(f"    → {out_path}  ({n_fixels} fixels)")
    return out_path


def copy_fixel_directory_files():
    """Copy index.mif and directions.mif from template fixel mask to fixel output dir."""
    for fname in ['index.mif', 'directions.mif']:
        src = os.path.join(TEMPLATE_FIXEL_MASK, fname)
        dst = os.path.join(FIXEL_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log.info(f"  Copied {fname} → {FIXEL_DIR}")
        else:
            log.warning(f"  NOT FOUND: {src} — fixel masks may not work without this")


def verify_all(net_names):
    """Final verification that all expected files exist and are non-empty."""
    log.info("\n=== VERIFICATION ===")
    all_ok = True
    for net in net_names:
        vox = os.path.join(VOXEL_DIR, f'{net}_voxel_mask.mif')
        fix = os.path.join(FIXEL_DIR, f'{net}_fixel_mask.mif')
        for path, label in [(vox, 'voxel'), (fix, 'fixel')]:
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                size_mb = os.path.getsize(path) / 1e6
                log.info(f"  OK  {net} {label}: {size_mb:.1f} MB")
            else:
                log.error(f"  MISSING or EMPTY: {path}")
                all_ok = False

    for fname in ['index.mif', 'directions.mif']:
        path = os.path.join(FIXEL_DIR, fname)
        if os.path.exists(path):
            log.info(f"  OK  {fname}")
        else:
            log.error(f"  MISSING: {path}")
            all_ok = False

    return all_ok


# MAIN 

def run_level_2():
    log.info("=" * 60)
    log.info("PHASE 8 — FINAL NETWORK MASK GENERATION")
    log.info(f"Output: {FINAL_DIR}")
    log.info("=" * 60)

    net_names = list(NETWORKS.keys())

    # Step 1: Copy fixel directory infrastructure
    log.info("\n[1/3] Copying fixel directory files...")
    copy_fixel_directory_files()

    # Step 2: Build voxel masks
    log.info("\n[2/3] Building voxel masks (3D binary)...")
    voxel_paths = {}
    for net_name, region_map in NETWORKS.items():
        log.info(f"\n  Network: {net_name}")
        voxel_paths[net_name] = build_voxel_mask(net_name, region_map)

    # Step 3: Build fixel masks
    log.info("\n[3/3] Converting to fixel space...")
    for net_name, voxel_path in voxel_paths.items():
        log.info(f"\n  Network: {net_name}")
        build_fixel_mask(net_name, voxel_path)

    # Cleanup tmp dir
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    # Final verification
    all_ok = verify_all(net_names)

    log.info("\n" + "=" * 60)
    if all_ok:
        log.info("SUCCESS — All masks generated and verified")
        log.info(f"\nFor DTI extraction, use:  {VOXEL_DIR}/{{NET}}_voxel_mask.mif")
        log.info(f"For FBA mrstats, use:     {FIXEL_DIR}/{{NET}}_fixel_mask.mif")
        log.info(f"For cfestats, use:        {FIXEL_DIR}/  (contains index + directions)")
    else:
        log.error("ONE OR MORE MASKS MISSING — check log above")
    log.info(f"\nLog: {LOG_FILE}")


if __name__ == "__main__":
    run_level_1()  # Level 1
    run_level_2()  # Level 2