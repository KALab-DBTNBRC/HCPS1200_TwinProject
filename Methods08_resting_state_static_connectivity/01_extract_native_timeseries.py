import os
import glob
import logging
import time
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1. CONFIGURATION & PATHS
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
DOWNLOADED_DIR = os.path.join(BASE_DIR, "Downloaded")
OUTPUT_TS_DIR = os.path.join(BASE_DIR, "Native_Timeseries")
ATLAS_PATH = os.path.join(BASE_DIR, "Atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S3.dlabel.nii")

# Number of CPU cores to use (Leaving 6 cores free so your PC doesn't freeze)
MAX_WORKERS = 30 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_TS_DIR, exist_ok=True)

# 2. LOGGING SETUP
log_file = os.path.join(BASE_DIR, f"Extraction_Master_Log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 3. GLOBAL ATLAS LOADER
# We load the atlas once to get the exact mapping of all 410 parcels
try:
    logging.info(f"Loading Master Atlas: {ATLAS_PATH}")
    atlas_img = nib.load(ATLAS_PATH)
    # The atlas data array contains the integer label for every single grayordinate
    atlas_data = atlas_img.get_fdata()[0] 
    
    # Get all unique integer labels in the atlas (excluding 0, which is the medial wall/background)
    PARCEL_LABELS = np.unique(atlas_data)
    PARCEL_LABELS = PARCEL_LABELS[PARCEL_LABELS > 0]
    
    if len(PARCEL_LABELS) != 410:
        logging.warning(f"Expected 410 parcels, but found {len(PARCEL_LABELS)}. Proceeding with extracted labels.")
    else:
        logging.info("Successfully validated 410 discrete parcels in the Atlas.")
        
except Exception as e:
    logging.error(f"FATAL: Could not load Atlas. {e}")
    exit()

# 4. CORE EXTRACTION FUNCTION (Worker)
def extract_subject_timeseries(subject_dir):
    sid = os.path.basename(subject_dir)
    
    # Find the specific dtseries file based on your nested structure
    file_pattern = os.path.join(subject_dir, "MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii")
    dtseries_files = glob.glob(file_pattern)
    
    if not dtseries_files:
        return sid, "FAILED", "Could not find dtseries.nii file."
    
    dtseries_path = dtseries_files[0]
    out_file = os.path.join(OUTPUT_TS_DIR, f"{sid}_native_timeseries.npy")
    
    # Skip if already processed (allows safe resuming if interrupted)
    if os.path.exists(out_file):
        return sid, "SKIPPED", "Already extracted."

    try:
        # Load the functional CIFTI natively into RAM
        img = nib.load(dtseries_path)
        
        # nibabel natively reads CIFTI as: [Timepoints, Grayordinates]
        data = img.get_fdata() 
        n_trs, n_grayordinates = data.shape
        
        # Sanity check: ensure the fMRI data aligns with the Atlas
        if n_grayordinates != len(atlas_data):
            return sid, "FAILED", f"Dimension Mismatch: fMRI has {n_grayordinates} grayordinates, Atlas has {len(atlas_data)}."

        # Initialize an empty matrix for the final timeseries [TRs x Parcels]
        ts_matrix = np.zeros((n_trs, len(PARCEL_LABELS)))
        
        # Extract the mean signal for every parcel
        for i, label_val in enumerate(PARCEL_LABELS):
            # Create a boolean mask where the atlas equals the current parcel
            mask = (atlas_data == label_val)
            # Calculate the mean across the spatial axis (axis=1)
            ts_matrix[:, i] = np.mean(data[:, mask], axis=1)
            
        # Check for NaNs
        if np.isnan(ts_matrix).any():
            return sid, "WARNING", f"Extracted {n_trs} TRs, but NaNs detected in timeseries."
            
        # Save as a blazing-fast numpy binary (.npy)
        np.save(out_file, ts_matrix)
        
        return sid, "SUCCESS", f"Extracted shape: {ts_matrix.shape} (TRs x Parcels)"
        
    except Exception as e:
        return sid, "FAILED", str(e)

# 5. PARALLEL EXECUTION ENGINE
def main():
    logging.info("Initializing Parallel Extraction Engine...")
    logging.info(f"Using {MAX_WORKERS} cores.")
    
    # Get list of subject directories
    subject_dirs = [d for d in glob.glob(os.path.join(DOWNLOADED_DIR, "*")) if os.path.isdir(d)]
    logging.info(f"Found {len(subject_dirs)} subjects in Downloaded directory.")
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs to the pool
        futures = {executor.submit(extract_subject_timeseries, sdir): os.path.basename(sdir) for sdir in subject_dirs}
        
        # Process as they finish
        for future in as_completed(futures):
            sid, status, msg = future.result()
            if status == "SUCCESS":
                logging.info(f"[{sid}] {status} - {msg}")
                success_count += 1
            elif status == "WARNING":
                logging.warning(f"[{sid}] {status} - {msg}")
                success_count += 1
            elif status == "SKIPPED":
                logging.info(f"[{sid}] {status} - {msg}")
                success_count += 1
            else:
                logging.error(f"[{sid}] {status} - {msg}")
                fail_count += 1

    elapsed = (time.time() - start_time) / 60
    logging.info("==========================================")
    logging.info("EXTRACTION COMPLETE")
    logging.info(f"Total Time: {elapsed:.2f} minutes")
    logging.info(f"Successful: {success_count} / {len(subject_dirs)}")
    logging.info(f"Failed:     {fail_count} / {len(subject_dirs)}")
    logging.info(f"Master Log saved to: {log_file}")
    logging.info("==========================================")

if __name__ == "__main__":
    main()