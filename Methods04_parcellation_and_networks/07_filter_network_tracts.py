import subprocess
import os
from multiprocessing import Pool

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

WHOLE_BRAIN_TCK = os.path.join(PROJECT_ROOT, "study_template/template_2M_filtered.tck")

BASE_MASK_DIR = os.path.join(PROJECT_ROOT, "network_masks")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "filtered_tracts")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NETWORKS = ["DMN", "Olfactory", "Reward", "Salience"]

def filter_network(net):
    """Function to be executed in parallel for each network."""
    mask_path = os.path.join(BASE_MASK_DIR, f"{net}_Network_bin.mif")
    output_tck = os.path.join(OUTPUT_DIR, f"{net.lower()}_network_filtered.tck")
    
    if not os.path.exists(mask_path):
        return f"[FAILED] Mask missing for {net} at: {mask_path}"

    cmd = [
        "tckedit", 
        WHOLE_BRAIN_TCK, 
        output_tck, 
        "-include", mask_path, 
        "-nthreads", "16", 
        "-force",
        "-quiet"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return f"[SUCCESS] {net} saved to {output_tck}"
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {net} extraction failed: {e}"

def run_parallel():
    print(f"Initializing Parallel Extraction on available cores...")
    print(f"Source: {WHOLE_BRAIN_TCK}")
    
    with Pool(processes=4) as pool:
        results = pool.map(filter_network, NETWORKS)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    run_parallel()