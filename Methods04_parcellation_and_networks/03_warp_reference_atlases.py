import subprocess
import os

# --- PATHS ---
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')
REF_DIR = os.path.join(BASE_DIR, 'reference_atlas')

# Inputs
STUDY_SCALAR_MIF = os.path.join(TEMPLATE_DIR, 'template_L0_3D.mif')
STUDY_SCALAR_NII = os.path.join(TEMPLATE_DIR, 'template_L0_3D.nii.gz')
MNI_T1 = os.path.join(os.environ.get("FSLDIR", "/usr/local/fsl"), "data/standard/MNI152_T1_1mm_brain.nii.gz")

# Glasser (Cortical), Tian (Subcortical), JHU (White Matter)
ATLASES = {
    'Glasser': os.path.join(REF_DIR, 'HCP-MMP/MNI_Glasser_HCP_v1.0.nii.gz'),
    'Tian': os.path.join(BASE_DIR, 'reference_atlas/Tian/Tian_Subcortex_S3_3T_1mm.nii.gz'),
    'JHU': os.path.join(BASE_DIR, 'reference_atlas/JHU-ICBM/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
}

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # 1. Convert Template to NIfTI
    print("\n--- STAGE 1: Preparing Template ---")
    run(['mrconvert', STUDY_SCALAR_MIF, STUDY_SCALAR_NII, '-force'])

    # 2. Generate the "Master Warp" (MNI -> Study)
    # Using Mutual Information to prevent NaN and SyN for edge-matching
    print("\n--- STAGE 2: Running ANTs Registration ---")
    prefix = os.path.join(TEMPLATE_DIR, 'MNI_to_Study_ANTs')
    run([
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', STUDY_SCALAR_NII,
        '-m', MNI_T1,
        '-o', prefix,
        '-t', 'sr', # SyN non-linear warp
        '-n', '30' # Adjust based on your server capacity
    ])

    # 3. Apply the Warp
    print("\n--- STAGE 3: Warping the Atlas Trio ---")
    warp = prefix + '1Warp.nii.gz'
    affine = prefix + '0GenericAffine.mat'

    for name, path in ATLASES.items():
        output = os.path.join(REF_DIR, f"{name}_in_Template_FINAL.nii.gz")
        run([
            'antsApplyTransforms',
            '-d', '3',
            '-i', path,
            '-r', STUDY_SCALAR_NII,
            '-o', output,
            '-n', 'NearestNeighbor', # Preserve label IDs
            '-t', warp,
            '-t', affine
        ])

    print("\n--- STAGE 4: Full Verification ---")
    print(f"Check all three in mrview:")
    print(f"mrview {STUDY_SCALAR_MIF} \\")
    print(f"  -overlay.load {REF_DIR}/Glasser_in_Template_FINAL.nii.gz \\")
    print(f"  -overlay.load {REF_DIR}/Tian_in_Template_FINAL.nii.gz \\")
    print(f"  -overlay.load {REF_DIR}/JHU_in_Template_FINAL.nii.gz")