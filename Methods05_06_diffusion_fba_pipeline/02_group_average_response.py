import os
import subprocess
import glob

# CONFIGURATION
BASE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "processed")
GROUP_RES_DIR = os.path.join(BASE_DIR, "group_responses")
os.makedirs(GROUP_RES_DIR, exist_ok=True)

def run_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {' '.join(cmd_list)}")
        return False

def run_phase_2():
    print("=== Phase 2: Generating Group Average Responses ===")
    
    # Locate all individual response files
    wm_files = glob.glob(os.path.join(BASE_DIR, "*", "*", "sub-*", "res_wm.txt"))
    gm_files = glob.glob(os.path.join(BASE_DIR, "*", "*", "sub-*", "res_gm.txt"))
    csf_files = glob.glob(os.path.join(BASE_DIR, "*", "*", "sub-*", "res_csf.txt"))

    if not wm_files:
        print("Error: No response files found. Check your paths.")
        return

    print(f"Found {len(wm_files)} subjects. Averaging...")

    # Create the Master Averages
    run_cmd(["responsemean"] + wm_files + [os.path.join(GROUP_RES_DIR, "master_wm.txt"), "-force"])
    run_cmd(["responsemean"] + gm_files + [os.path.join(GROUP_RES_DIR, "master_gm.txt"), "-force"])
    run_cmd(["responsemean"] + csf_files + [os.path.join(GROUP_RES_DIR, "master_csf.txt"), "-force"])
    
    print(f" Success! Master responses created in {GROUP_RES_DIR}")

if __name__ == "__main__":
    run_phase_2()