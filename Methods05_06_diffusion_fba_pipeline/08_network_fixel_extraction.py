import os
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
STATS_BASE_DIR = os.path.join(BASE_DIR, 'stats', 'cfestats_ACT_SIFT2_filtered')
MASK_DIR = os.path.join(BASE_DIR, 'network_masks') 
OUT_DIR = os.path.join(BASE_DIR, 'stats', 'newnetwork_stats', 'exploratory_uncorrected')

os.makedirs(OUT_DIR, exist_ok=True)

# Define the exact folder naming conventions you established
METRICS = ['fdc', 'fd', 'fc']
DIRECTIONS = ['loss', 'gain']

# Your 1D Fixel Masks
NETWORK_NAMES = [
    "Reward_Network_bin",
    "Salience_Network_bin",
    "DMN_Network_bin",
    "Olfactory_Network_bin"
]

def exploratory_stats():
    print(f"[{datetime.now()}] Starting Uncorrected Exploratory Extraction...")
    print("Evaluating sub-threshold clustering (p < 0.05 uncorrected) across all FBA metrics and directions.")
    print("=" * 70)

    # Iterate through the metrics (fc, fd, fdc)
    for metric in METRICS:
        # Iterate through the directions (loss, gain)
        for direction in DIRECTIONS:
            
            # Construct the nested path based on your exact structure
            stats_folder = f"{metric}stats"
            results_folder = f"results_{metric}_{direction}"
            stats_file = os.path.join(STATS_BASE_DIR, stats_folder, results_folder, 'uncorrected_pvalue.mif')
            
            if not os.path.exists(stats_file):
                print(f"WARNING: Cannot find {stats_file}. Skipping...")
                continue
                
            print(f"\n>>> ANALYZING METRIC: {metric.upper()} | DIRECTION: {direction.upper()} <<<")
            print("-" * 60)

            for name in NETWORK_NAMES:
                mask_path = os.path.join(MASK_DIR, "networks_1D_format", f"{name}_1D.mif")
                
                # Create a highly specific output filename so nothing gets overwritten
                sig_out_path = os.path.join(OUT_DIR, f"uncorrected_sig_{metric}_{direction}_{name}.mif")

                # STEP 1: INTERSECTION
                # Multiply the uncorrected p-value > 0.95 map by the 1D network fixel mask
                calc_cmd = [
                    "mrcalc", stats_file, "0.95", "-gt", 
                    mask_path, "-mult", sig_out_path, "-force", "-quiet"
                ]
                subprocess.run(calc_cmd, check=True)

                # STEP 2: COUNTING
                total_count = subprocess.check_output(["mrstats", mask_path, "-output", "count"]).decode().strip()
                sig_count = subprocess.check_output(["mrstats", sig_out_path, "-output", "count", "-ignorezero"]).decode().strip()

                # Handle empty outputs gracefully
                sig_count = 0 if not sig_count else int(float(sig_count))
                total_count = 0 if not total_count else int(float(total_count))

                # STEP 3: RESULTS OUTPUT
                percentage = (sig_count / total_count) * 100 if total_count > 0 else 0
                
                clean_name = name.replace('_fixel', ' Network')
                print(f"NETWORK: {clean_name}")
                print(f"  > Total size: {total_count} fixels")
                print(f"  > Hits (p<0.05 uncorrected): {sig_count} fixels")
                print(f"  > Network Density:  {percentage:.2f}%")
                print("-" * 60)

if __name__ == "__main__":
    exploratory_stats()