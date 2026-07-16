import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import warnings

warnings.filterwarnings("ignore")

def run_network_chronnectome():
    # ==========================================
    # 1. CONFIGURATION & PATHS (CORRECTED)
    # ==========================================
    BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
    TS_DIR = os.path.join(BASE_DIR, "Network_Timeseries") 
    OUTPUT_DIR = os.path.join(BASE_DIR, "Tables/Phase4_Dynamics")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- SLIDING WINDOW PARAMETERS ---
    WINDOW_SIZE_TR = 30  # Adjust if your TR makes 30 too short/long (aim for ~45 secs)
    STEP_SIZE_TR = 2     

    print(f"[{time.strftime('%H:%M:%S')}] INITIATING PHASE 4A: NETWORK-LEVEL DYNAMICS")
    
    if not os.path.exists(TS_DIR):
        print(f"[!] ERROR: Directory not found at {TS_DIR}")
        return

    # Look specifically for the Network CSVs
    subject_files = sorted([f for f in os.listdir(TS_DIR) if f.endswith('_4_networks.csv')])
    if not subject_files:
        print(f"[!] ERROR: No '_4_networks.csv' files found in {TS_DIR}.")
        return

    # ==========================================
    # 2. GENERATE SLIDING WINDOWS & FISHER-Z
    # ==========================================
    all_windows = []
    subject_window_counts = {}
    subject_ids = []
    edge_names = [] # To keep track of what the 6 edges actually are

    print(f"[{time.strftime('%H:%M:%S')}] Generating dynamic networks for {len(subject_files)} subjects...")

    for i, f in enumerate(subject_files):
        # Extract ID (e.g., '100610' from '100610_4_networks.csv')
        sub_id = f.split('_')[0]
        file_path = os.path.join(TS_DIR, f)
        
        try:
            df = pd.read_csv(file_path)
            
            # Drop any non-numeric columns (like 'Time' or 'TR' indices)
            df_numeric = df.select_dtypes(include=[np.number])
            
            # Grab column names on the first pass to define the edge pairs
            if i == 0:
                cols = df_numeric.columns.tolist()
                for row_idx in range(len(cols)):
                    for col_idx in range(row_idx + 1, len(cols)):
                        edge_names.append(f"{cols[row_idx]}-{cols[col_idx]}")
                print(f"  > Detected Networks: {cols}")
                print(f"  > Tracking {len(edge_names)} Dynamic Edges: {edge_names}")

            ts_data = df_numeric.values
            n_vols = ts_data.shape[0]
            sub_windows = []
            
            for start in range(0, n_vols - WINDOW_SIZE_TR + 1, STEP_SIZE_TR):
                end = start + WINDOW_SIZE_TR
                window_ts = ts_data[start:end, :]
                
                # Correlation matrix for the 4 networks
                corr_mat = np.corrcoef(window_ts, rowvar=False)
                
                # Extract the 6 upper-triangle edges
                triu_indices = np.triu_indices_from(corr_mat, k=1)
                edges = corr_mat[triu_indices]
                
                # Fisher Z-Transform with clipping
                edges_clipped = np.clip(edges, -0.9999, 0.9999)
                z_edges = np.arctanh(edges_clipped)
                
                sub_windows.append(z_edges)
            
            if len(sub_windows) > 0:
                sub_windows_arr = np.vstack(sub_windows)
                if np.isnan(sub_windows_arr).any():
                    sub_windows_arr = np.nan_to_num(sub_windows_arr, nan=0.0)
                    
                all_windows.append(sub_windows_arr)
                subject_window_counts[sub_id] = sub_windows_arr.shape[0]
                subject_ids.append(sub_id)
                
        except Exception as e:
            print(f"  [!] Failed to process {f}: {e}")

    pooled_data = np.vstack(all_windows)
    print(f"[{time.strftime('%H:%M:%S')}] Total temporal windows pooled: {pooled_data.shape[0]}")

    # ==========================================
    # 3. SILHOUETTE OPTIMIZATION (k=2 to 5)
    # ==========================================
    print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating optimal states...")
    best_k = 2
    best_score = -1
    
    # We can use the whole dataset here because 6 features is incredibly fast to compute
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pooled_data)
        
        # Taking a large sample to keep it fast but highly accurate
        sample_size = min(30000, pooled_data.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(pooled_data.shape[0], sample_size, replace=False)
        
        score = silhouette_score(pooled_data[sample_idx], labels[sample_idx])
        print(f"  > k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n[WINNER] k={best_k} (Score: {best_score:.4f})")

    # ==========================================
    # 4. FINAL CLUSTERING & EXTRACTION
    # ==========================================
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    global_labels = final_kmeans.fit_predict(pooled_data)

    # Save Centroids
    centroid_path = os.path.join(OUTPUT_DIR, f"dFC_Centroids_k{best_k}.npy")
    np.save(centroid_path, final_kmeans.cluster_centers_)
    
    # Save the Edge Names mapping so you can interpret the centroids
    edge_mapping_df = pd.DataFrame({'Edge_Index': range(len(edge_names)), 'Edge_Name': edge_names})
    edge_mapping_df.to_csv(os.path.join(OUTPUT_DIR, "dFC_Edge_Mapping.csv"), index=False)

    subject_metrics = []
    current_idx = 0

    for sub_id in subject_ids:
        n_win = subject_window_counts[sub_id]
        sub_labels = global_labels[current_idx : current_idx + n_win]
        current_idx += n_win
        
        dwell_times = {f"Dwell_State_{i+1}": np.sum(sub_labels == i) / n_win for i in range(best_k)}
        transitions = np.sum(np.diff(sub_labels) != 0)
        dominant = np.argmax([dwell_times[f"Dwell_State_{i+1}"] for i in range(best_k)]) + 1
        
        row = {'Subject': sub_id, 'Total_Transitions': transitions, 'Dominant_State': dominant}
        row.update(dwell_times)
        subject_metrics.append(row)

    df_metrics = pd.DataFrame(subject_metrics)
    out_csv = os.path.join(OUTPUT_DIR, f"dFC_Subject_Metrics_k{best_k}.csv")
    df_metrics.to_csv(out_csv, index=False)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] SUCCESS!")
    print(f"Saved Metrics: {out_csv}")
    print(f"Saved Edge Legend: dFC_Edge_Mapping.csv")

if __name__ == "__main__":
    run_network_chronnectome()