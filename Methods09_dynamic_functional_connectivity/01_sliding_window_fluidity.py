import os
import glob
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
INPUT_DIR = os.path.join(BASE_DIR, "Native_Timeseries")
OUTPUT_CSV = os.path.join(BASE_DIR, "Tables/rsfMRI_Tier3_Dynamic_Metrics.csv")

# Sliding Window Parameters
WINDOW_SIZE = 100  # 100 TRs (~72 seconds)
STEP_SIZE = 20     # Slide window by 20 TRs (~14 seconds)
MAX_WORKERS = 30

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

# FULL TOPOLOGICAL ALIGNMENT MAP
network_indices = {
    'Reward': [140, 320, 122, 302, 138, 318, 115, 295, 141, 321, 142, 322, 143, 323, 216, 396, 214, 394, 215, 395, 161, 341, 162, 342, 111, 291, 230, 410, 22, 47, 23, 48, 15, 40, 16, 41, 17, 42, 18, 43, 11, 36, 12, 37, 13, 38, 14, 39, 19, 44, 20, 45],
    'Salience': [161, 341, 162, 342, 159, 339, 107, 287, 109, 289, 90, 270, 129, 309, 130, 310, 131, 311, 163, 343, 165, 345, 164, 344, 19, 44, 20, 45, 9, 34],
    'DMN': [211, 391, 212, 392, 85, 265, 83, 263, 84, 264, 64, 244, 115, 295, 122, 302, 138, 318, 77, 257, 80, 260, 200, 380, 193, 373, 178, 358, 176, 356, 177, 357, 205, 385, 1, 26, 2, 27, 3, 28, 4, 29],
    'Olfactory': [160, 340, 168, 348, 169, 349, 181, 361, 222, 402, 143, 323, 216, 396, 142, 322, 214, 394, 19, 44, 20, 45, 1, 26, 2, 27, 3, 28, 4, 29]
}

# DYNAMIC EXTRACTION WORKER
def process_dynamic_fc(filepath):
    sid = os.path.basename(filepath).split('_')[0]
    try:
        ts_410 = np.load(filepath)
        n_trs, n_parcels = ts_410.shape
        
        n_windows = (n_trs - WINDOW_SIZE) // STEP_SIZE
        if n_windows <= 0:
            return sid, "FAILED", "Timeseries too short for window parameters."
            
        n_edges = (n_parcels * (n_parcels - 1)) // 2
        edge_dynamics = np.zeros((n_windows, n_edges))

        # 1. SLIDING WINDOW CORRELATIONS
        for w in range(n_windows):
            start = w * STEP_SIZE
            end = start + WINDOW_SIZE
            window_data = ts_410[start:end, :]
            
            corr_mat = np.corrcoef(window_data.T)
            z_mat = fisher_z(corr_mat)
            
            # Store only the upper triangle to save massive amounts of RAM
            edge_dynamics[w, :] = z_mat[np.triu_indices(n_parcels, k=1)]
            
        # EDGE VARIANCE CALCULATION
        # Calculate how much each specific edge changes across all windows
        edge_variances = np.var(edge_dynamics, axis=0)
        global_fluidity = np.mean(edge_variances)
        
        res = {
            'Subject': int(sid),
            'Global_Dynamic_Fluidity': global_fluidity,
            'Window_Count': n_windows
        }
        
        # NETWORK-SPECIFIC FLUIDITY
        # We need to map the flat edge_variances back to a 410x410 structure
        var_matrix = np.zeros((n_parcels, n_parcels))
        var_matrix[np.triu_indices(n_parcels, k=1)] = edge_variances
        var_matrix = var_matrix + var_matrix.T # Make symmetric
        
        for net_name, ids in network_indices.items():
            valid_idx = [idx - 1 for idx in ids if 0 <= (idx - 1) < 410]
            
            # Extract the variance matrix for JUST this network
            within_var_submat = var_matrix[np.ix_(valid_idx, valid_idx)]
            
            # We only want the unique connections within the network (upper triangle)
            # k=1 excludes the diagonal (variance of a node with itself is meaningless)
            within_vars = within_var_submat[np.triu_indices(len(valid_idx), k=1)]
            
            # Mean variance of the connections within this specific network
            res[f'Dynamic_Fluidity_{net_name}'] = np.nanmean(within_vars)

        return sid, "SUCCESS", res
        
    except Exception as e:
        return sid, "FAILED", str(e)

# EXECUTION POOL
if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] Initializing Phase 4A: Dynamic Fluidity Extraction...")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    print(f"Processing {len(files)} subjects using {MAX_WORKERS} cores...")
    
    if not files:
        print("ERROR: No .npy files found in INPUT_DIR.")
        exit()
        
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_dynamic_fc, f): f for f in files}
        for future in as_completed(futures):
            sid, status, output = future.result()
            if status == "SUCCESS":
                results.append(output)
            else:
                print(f"[{sid}] FAILED: {output}")

    # Save Dynamic Metrics
    if results:
        df_dyn = pd.DataFrame(results)
        df_dyn.to_csv(OUTPUT_CSV, index=False)
        
        elapsed = (time.time() - start_time) / 60
        print(f"--- PHASE 4A COMPLETE in {elapsed:.2f} min ---")
        print(f"Saved Dynamic Metrics: {OUTPUT_CSV}")
    else:
        print("ERROR: No results were successfully generated.")