import os
import numpy as np
import pandas as pd
import glob
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATION 
GLM_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "GLM_Contrasts")
OUT_DIR = os.environ.get("PROJECT_ROOT", ".")
LOG_FILE = os.path.join(os.path.join(os.environ.get("PROJECT_ROOT", "."), "logs"), f"Global_tfMRI_Audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
SUMMARY_FILE = os.path.join(os.path.join(os.environ.get("PROJECT_ROOT", "."), "tables"), "Global_tfMRI_Sanity_Summary.csv")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# VERIFIED ATLAS ROI DEFINITIONS (0-Based Python Indices)
ROI_MAP = {
    # REWARD NETWORK
    'GAMBLING_win_anticipation': {'name': 'NAc', 'indices': [21, 22, 46, 47]},
    'GAMBLING_loss_anticipation': {'name': 'Anterior Insula (Salience)', 'indices': [160, 161, 340, 341]},
    'GAMBLING_reward_loss_diff': {'name': 'NAc', 'indices': [21, 22, 46, 47]},
    'GAMBLING_reward_block': {'name': 'NAc', 'indices': [21, 22, 46, 47]},
    
    # SALIENCE / AMYGDALA
    'EMOTION_fear_response': {'name': 'Amygdala', 'indices': [18, 19, 43, 44]},
    
    # DEFAULT MODE NETWORK
    'SOCIAL_theory_of_mind': {'name': 'mPFC (10r)', 'indices': [114, 294]},
    
    # EXECUTIVE / FRONTOPARIETAL
    'WM_executive_load': {'name': 'IFJ (Salience/Exec)', 'indices': [128, 129, 308, 309]},
    'WM_face_executive': {'name': 'IFJ (Salience/Exec)', 'indices': [128, 129, 308, 309]},
    
    # ANTERIOR PFC / REASONING
    'RELATIONAL_relational_reasoning': {'name': 'Frontal Pole (10pp)', 'indices': [139, 319]},
    
    # LANGUAGE / TEMPORAL
    'LANGUAGE_narrative': {'name': 'Temporal Pole / Auditory', 'indices': [180, 221, 360, 401]},
    
    # MOTOR / SOMATOMOTOR
    # Broad check for somatomotor variance rather than a single digit
    'MOTOR_motor_control': {'name': 'Global Cortical Variance', 'indices': list(range(0, 360))}
}

#  EXECUTION 
if __name__ == "__main__":
    logging.info(" STARTING GLOBAL tfMRI DATA INTEGRITY & SANITY AUDIT ")
    
    all_results = []
    corruption_flags = 0
    
    # Get all unique tasks and contrasts present in the directory
    all_files = glob.glob(os.path.join(GLM_DIR, "*.npy"))
    if not all_files:
        logging.error(f"FATAL: No .npy files found in {GLM_DIR}")
        exit()
        
    logging.info(f"Located {len(all_files)} total GLM contrast files. Beginning sweep...\n")

    for task_con, config in ROI_MAP.items():
        task, contrast = task_con.split('_', 1)
        search_pattern = os.path.join(GLM_DIR, f"*_{task}_{contrast}.npy")
        files = glob.glob(search_pattern)
        
        if not files:
            logging.warning(f"Skipping {task_con}: No files found.")
            continue
            
        logging.info(f"Auditing [ {task_con} ] (n={len(files)})...")
        
        task_betas = []
        for f in files:
            sub_id = os.path.basename(f).split('_')[0]
            try:
                data = np.load(f)
            except Exception as e:
                logging.error(f"CORRUPTION: Could not load {os.path.basename(f)}. Error: {e}")
                corruption_flags += 1
                continue
            
            #  INTEGRITY CHECKS 
            if data.shape != (410,):
                logging.error(f"DIMENSION MISMATCH: {sub_id} has shape {data.shape}, expected (410,)")
                corruption_flags += 1
                continue
            if np.isnan(data).any() or np.isinf(data).any():
                logging.error(f"MATH ERROR: {sub_id} contains NaNs or Infs.")
                corruption_flags += 1
                continue
            if np.std(data) < 1e-6:
                logging.error(f"FLAT SIGNAL: {sub_id} has near-zero variance.")
                corruption_flags += 1
                continue
                
            #  BIOLOGICAL SANITY CHECK 
            roi_beta = np.mean(data[config['indices']])
            task_betas.append(roi_beta)
            all_results.append({
                'Subject': sub_id,
                'Task': task,
                'Contrast': contrast,
                'Target_ROI': config['name'],
                'Mean_Beta': roi_beta,
                'Status': 'PASS'
            })
        
        # Report group-level biological sanity for this contrast
        if task_betas:
            mean_val = np.mean(task_betas)
            std_val = np.std(task_betas)
            pos_pct = (np.array(task_betas) > 0).mean() * 100
            
            logging.info(f"    Target Region: {config['name']}")
            logging.info(f"    Group Mean Activation: {mean_val:.4f} (±{std_val:.4f})")
            logging.info(f"    Spatial Consistency: {pos_pct:.1f}% subjects > 0")
            
            if pos_pct >= 75:
                logging.info(f"    VERDICT: EXCELLENT (Robust biological mapping)\n")
            elif pos_pct >= 50:
                logging.info(f"    VERDICT: ACCEPTABLE (Expected noise within population)\n")
            else:
                logging.warning(f"    VERDICT: WEAK (Task did not strongly drive targeted network)\n")

    # Save flat table
    df = pd.DataFrame(all_results)
    df.to_csv(SUMMARY_FILE, index=False)
    
    logging.info("-" * 50)
    logging.info(" AUDIT COMPLETE ")
    logging.info(f"Total Valid Files Processed: {len(df)}")
    logging.info(f"Total Corrupted/Failed Files: {corruption_flags}")
    if corruption_flags == 0:
        logging.info("DATA INTEGRITY VERIFIED: Ready for Phase 1 Landscape Synthesis.")
    logging.info(f"Results saved to {SUMMARY_FILE}")