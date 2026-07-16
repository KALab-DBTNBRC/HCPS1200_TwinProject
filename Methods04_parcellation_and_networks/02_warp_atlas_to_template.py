import os
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
TEMPLATE_DIR = os.path.join(BASE_DIR, 'study_template')
ATLAS_DIR = os.path.join(BASE_DIR, 'ReferenceAtlas')

# Warps and Targets
WARP_FILE = os.path.join(TEMPLATE_DIR, 'mni_to_template_warp.mif')
MY_TEMPLATE = os.path.join(TEMPLATE_DIR, 'template_L0_3D.mif') # Use the 3D version

# Input Atlases
GLASSER_MNI = os.path.join(ATLAS_DIR, 'glasser_3D.mif')
DK_MNI = os.path.join(ATLAS_DIR, 'Desikan_space-MNI152NLin6_res-1x1x1.nii')

# Outputs
GLASSER_TPL = os.path.join(TEMPLATE_DIR, 'glasser_on_template.mif')
DK_TPL = os.path.join(TEMPLATE_DIR, 'dk_on_template.mif')

def warp_atlases():
    print(f"[{datetime.now()}] Warping Atlases to Template Space...")
    for src, out in [(GLASSER_MNI, GLASSER_TPL), (DK_MNI, DK_TPL)]:
        # Using -interp nearest to keep integer label IDs intact
        cmd = [
            "mrtransform", src, "-warp", WARP_FILE, 
            "-interp", "nearest", "-template", MY_TEMPLATE, 
            out, "-force"
        ]
        subprocess.run(cmd, check=True)
        print(f"  -> Generated: {os.path.basename(out)}")

if __name__ == "__main__":
    warp_atlases()