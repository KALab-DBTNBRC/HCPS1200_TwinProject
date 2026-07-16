import os
import sys
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = os.environ.get("PROJECT_ROOT", ".")
MIND_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "MIND_matrices")
CSV_PATH = os.path.join(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/Twins_240_beh_sheet_complete_all_vars - Sheet1.csv"))

OUTPUT_DIR = os.path.join(PROJECT_DIR, "GroupAvgValidation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 75

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("MIND_Phase2_MaxIO")

def load_single_matrix(sub):
    """Worker function to read a single numpy array from disk."""
    npy_path = os.path.join(MIND_DIR, f"{sub}_MIND.npy")
    if os.path.exists(npy_path):
        return sub, np.load(npy_path)
    return sub, None

def main():
    logger.info("=== PHASE 2: HIGH-THROUGHPUT MIND AGGREGATION ===")

    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        logger.critical(f"Could not find twin table at {CSV_PATH}")
        sys.exit(1)

    df_clean = df[df['TwinPairID'] != 'Pair41'].dropna(subset=['Subject']).copy()
    expected_subjects = [str(int(float(s))) for s in df_clean['Subject'].values]
    
    logger.info(f"Target Demographic Table: {len(expected_subjects)} subjects expected.")

    logger.info(f"Igniting ThreadPoolExecutor with {MAX_WORKERS} concurrent I/O streams...")
    matrices_dict = {}
    missing_subjects = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(load_single_matrix, sub): sub for sub in expected_subjects}
        
        for future in as_completed(futures):
            sub, mat = future.result()
            if mat is not None:
                matrices_dict[sub] = mat
            else:
                missing_subjects.append(sub)

    if missing_subjects:
        logger.error(f"Missing matrices for {len(missing_subjects)} subjects: {missing_subjects}")
        sys.exit(1)

    logger.info(f"I/O Complete. Successfully loaded {len(matrices_dict)} MIND matrices into RAM.")

    logger.info("Computing Group Average Network (Vectorized)...")
    stack = np.stack([matrices_dict[sub] for sub in expected_subjects], axis=0)  
    group_matrix = np.mean(stack, axis=0)

    logger.info("Extracting Nodal Degrees (Hubs)...")
    nodal_degrees = np.sum(group_matrix, axis=1)

    sample_csv = os.path.join(MIND_DIR, f"{expected_subjects[0]}_MIND.csv")
    region_names = pd.read_csv(sample_csv, index_col=0).columns.tolist()

    group_npy_path = os.path.join(OUTPUT_DIR, "Group_Average_MIND.npy")
    group_csv_path = os.path.join(OUTPUT_DIR, "Group_Average_MIND.csv")
    hub_csv_path = os.path.join(OUTPUT_DIR, "Group_MIND_Nodal_Degrees.csv")

    np.save(group_npy_path, group_matrix)
    pd.DataFrame(group_matrix, index=region_names, columns=region_names).to_csv(group_csv_path)
    
    df_hubs = pd.DataFrame({'Region': region_names, 'Weighted_Degree': nodal_degrees})
    df_hubs.to_csv(hub_csv_path, index=False)

    mean_connectivity = np.mean(group_matrix[~np.eye(360, dtype=bool)])
    max_hub = df_hubs.loc[df_hubs['Weighted_Degree'].idxmax()]
    min_hub = df_hubs.loc[df_hubs['Weighted_Degree'].idxmin()]

    logger.info("=== VALIDATION SUMMARY ===")
    logger.info(f"Group Matrix Shape: {group_matrix.shape}")
    logger.info(f"Mean Off-Diagonal Similarity: {mean_connectivity:.4f}")
    logger.info(f"Most Connected Hub: {max_hub['Region']} (Degree: {max_hub['Weighted_Degree']:.2f})")
    logger.info(f"Least Connected Node: {min_hub['Region']} (Degree: {min_hub['Weighted_Degree']:.2f})")
    logger.info(f"All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
