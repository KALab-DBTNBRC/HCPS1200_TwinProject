import os
import glob
import logging
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION & PATHS
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
INPUT_DIR = os.path.join(BASE_DIR, "Native_Timeseries")
OUTPUT_DIR = os.path.join(BASE_DIR, "Network_Timeseries")

MAX_WORKERS = 30
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(BASE_DIR, f"Phase2_Network_Alignment_Log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# THE TOPOLOGICAL ALIGNMENT MAP
# Verified mapping from Glasser/Tian strings to 1-based integer labels
network_indices = {
    'Reward': [
        # Prefrontal / Frontal Polar
        140, # R_10pp_ROI
        320, # L_10pp_ROI
        122, # R_10d_ROI
        302, # L_10d_ROI
        138, # R_10v_ROI
        318, # L_10v_ROI
        115, # R_10r_ROI
        295, # L_10r_ROI
        
        # Orbitofrontal
        141, # R_11l_ROI
        321, # L_11l_ROI
        142, # R_13l_ROI
        322, # L_13l_ROI
        143, # R_OFC_ROI
        323, # L_OFC_ROI
        216, # R_pOFC_ROI
        396, # L_pOFC_ROI
        
        # Subcallosal / ACC
        214, # R_25_ROI
        394, # L_25_ROI
        215, # R_s32_ROI
        395, # L_s32_ROI
        111, # R_a24_ROI
        291, # L_a24_ROI
        230, # R_p24_ROI
        410, # L_p24_ROI
        
        # Insula (Reward-relevant portions)
        161, # R_AVI_ROI
        341, # L_AVI_ROI
        162, # R_AAIC_ROI
        342, # L_AAIC_ROI
        
        # Striatum (Tian)
        22,  # NAc-shell-rh
        47,  # NAc-shell-lh
        23,  # NAc-core-rh
        48,  # NAc-core-lh
        15,  # CAU-VA-rh
        40,  # CAU-VA-lh
        16,  # CAU-DA-rh
        41,  # CAU-DA-lh
        17,  # CAU-body-rh
        42,  # CAU-body-lh
        18,  # CAU-tail-rh
        43,  # CAU-tail-lh
        11,  # PUT-VA-rh
        36,  # PUT-VA-lh
        12,  # PUT-DA-rh
        37,  # PUT-DA-lh
        13,  # PUT-VP-rh
        38,  # PUT-VP-lh
        14,  # PUT-DP-rh
        39,  # PUT-DP-lh
        
        # Amygdala (Tian)
        19,  # lAMY-rh
        44,  # lAMY-lh
        20,  # mAMY-rh
        45   # mAMY-lh
    ],
    
    'Salience': [
        # Core Insula
        161, # R_AVI_ROI
        341, # L_AVI_ROI
        162, # R_AAIC_ROI
        342, # L_AAIC_ROI
        159, # R_MI_ROI
        339, # L_MI_ROI
        
        # Dorsal ACC
        107, # R_p24pr_ROI
        287, # L_p24pr_ROI
        109, # R_a24pr_ROI
        289, # L_a24pr_ROI
        90,  # R_24dd_ROI
        270, # L_24dd_ROI
        
        # Inferior Frontal Junction
        129, # R_IFJa_ROI
        309, # L_IFJa_ROI
        130, # R_IFJp_ROI
        310, # L_IFJp_ROI
        131, # R_IFSp_ROI
        311, # L_IFSp_ROI
        
        # Frontal Operculum
        163, # R_FOP1_ROI
        343, # L_FOP1_ROI
        165, # R_FOP2_ROI
        345, # L_FOP2_ROI
        164, # R_FOP3_ROI
        344, # L_FOP3_ROI
        
        # Subcortex (Tian)
        19,  # lAMY-rh
        44,  # lAMY-lh
        20,  # mAMY-rh
        45,  # mAMY-lh
        9,   # THA-DAm-rh
        34   # THA-DAm-lh
    ],
    
    'DMN': [
        # Posterior Cingulate
        211, # R_31pd_ROI
        391, # L_31pd_ROI
        212, # R_31a_ROI
        392, # L_31a_ROI
        85,  # R_31pv_ROI
        265, # L_31pv_ROI
        83,  # R_v23ab_ROI
        263, # L_v23ab_ROI
        84,  # R_d23ab_ROI
        264, # L_d23ab_ROI
        64,  # R_RSC_ROI
        244, # L_RSC_ROI
        
        # Medial Prefrontal
        115, # R_10r_ROI
        295, # L_10r_ROI
        122, # R_10d_ROI
        302, # L_10d_ROI
        138, # R_10v_ROI
        318, # L_10v_ROI
        
        # Precuneus
        77,  # R_PCV_ROI
        257, # L_PCV_ROI
        80,  # R_7m_ROI
        260, # L_7m_ROI
        
        # Parietal / Angular Gyrus
        200, # R_PGi_ROI
        380, # L_PGi_ROI
        193, # R_PGp_ROI
        373, # L_PGp_ROI
        
        # Temporal
        178, # R_STSda_ROI
        358, # L_STSda_ROI
        
        # Parahippocampal
        176, # R_PHA1_ROI
        356, # L_PHA1_ROI
        177, # R_PHA3_ROI
        357, # L_PHA3_ROI
        205, # R_PHA2_ROI
        385, # L_PHA2_ROI
        
        # Hippocampus (Tian)
        1,   # HIP-head-m-rh
        26,  # HIP-head-m-lh
        2,   # HIP-head-l-rh
        27,  # HIP-head-l-lh
        3,   # HIP-body-rh
        28,  # HIP-body-lh
        4,   # HIP-tail-rh
        29   # HIP-tail-lh
    ],
    
    'Olfactory': [
        # Primary Olfactory / Cortex
        160, # R_Pir_ROI
        340, # L_Pir_ROI
        168, # R_EC_ROI
        348, # L_EC_ROI
        169, # R_PreS_ROI
        349, # L_PreS_ROI
        
        # Temporal Pole
        181, # R_TGd_ROI
        361, # L_TGd_ROI
        222, # R_TGv_ROI
        402, # L_TGv_ROI
        
        # Orbitofrontal / Subcallosal (Medial Tract Routing)
        143, # R_OFC_ROI
        323, # L_OFC_ROI
        216, # R_pOFC_ROI
        396, # L_pOFC_ROI
        142, # R_13l_ROI
        322, # L_13l_ROI
        214, # R_25_ROI
        394, # L_25_ROI
        
        # Amygdala (Tian - Direct Bulb Input)
        19,  # lAMY-rh
        44,  # lAMY-lh
        20,  # mAMY-rh
        45,  # mAMY-lh
        
        # Hippocampus (Tian - Olfactory Memory)
        1,   # HIP-head-m-rh
        26,  # HIP-head-m-lh
        2,   # HIP-head-l-rh
        27,  # HIP-head-l-lh
        3,   # HIP-body-rh
        28,  # HIP-body-lh
        4,   # HIP-tail-rh
        29   # HIP-tail-lh
    ]
}

# WORKER FUNCTION
def map_networks(npy_file):
    filename = os.path.basename(npy_file)
    sid = filename.split('_')[0]
    
    out_csv = os.path.join(OUTPUT_DIR, f"{sid}_4_networks.csv")
    
    if os.path.exists(out_csv):
        return sid, "SKIPPED", "Network CSV already exists."

    try:
        # Load the [4800 x 410] Native Timeseries
        ts_410 = np.load(npy_file)
        n_trs, n_parcels = ts_410.shape
        
        if n_parcels != 410:
            return sid, "FAILED", f"Expected 410 parcels, but found {n_parcels}."

        # Compute Custom Network Timeseries
        net_data = {}
        for net_name, ids in network_indices.items():
            # Convert 1-based Atlas IDs to 0-based Python indices
            valid_idx = [idx - 1 for idx in ids if 0 <= (idx - 1) < 410]
            
            net_ts_matrix = ts_410[:, valid_idx]
            net_data[net_name] = np.mean(net_ts_matrix, axis=1)
            
        # Save to CSV
        df_net = pd.DataFrame(net_data)
        df_net.to_csv(out_csv, index=False)
        
        return sid, "SUCCESS", f"Aligned {len(network_indices)} networks."
        
    except Exception as e:
        return sid, "FAILED", str(e)

# EXECUTION POOL
if __name__ == "__main__":
    logging.info("Initializing Phase 2 Topological Alignment...")
    
    npy_files = glob.glob(os.path.join(INPUT_DIR, "*.npy"))
    logging.info(f"Found {len(npy_files)} native timeseries files.")
    
    if len(npy_files) == 0:
        logging.error("No .npy files found. Please check your INPUT_DIR path.")
        exit()
    
    start_time = time.time()
    success, fail = 0, 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(map_networks, f): os.path.basename(f) for f in npy_files}
        
        for future in as_completed(futures):
            sid, status, msg = future.result()
            if status == "SUCCESS" or status == "SKIPPED":
                success += 1
            else:
                logging.error(f"[{sid}] {status} - {msg}")
                fail += 1

    elapsed = (time.time() - start_time) / 60
    logging.info("--- PHASE 2 ALIGNMENT COMPLETE ---")
    logging.info(f"Time: {elapsed:.2f} min | OK: {success} | Failed: {fail}")