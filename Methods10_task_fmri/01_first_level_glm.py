import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import logging
import datetime

warnings.filterwarnings('ignore')

#  CONFIGURATION 
SCRIPT_VERSION = "3.0.0_Unified_GLM"
MAX_WORKERS = 65 
TR = 0.72

BASE_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "Organized")
OUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "GLM_Contrasts")
AUDIT_FILE = os.path.join(os.environ.get("PROJECT_ROOT", "."), "tables/QC_Audit_Report.csv")
LOG_FILE = os.path.join(os.path.join(os.environ.get("PROJECT_ROOT", "."), "logs"), f"Phase0D_GLM_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

#  CONTRAST DICTIONARY 
# Defines the specific EV files to load and the mathematical contrasts to compute
TASK_CONFIG = {
    "GAMBLING": {
        "evs": ['win', 'loss', 'win_event', 'loss_event', 'neut_event'],
        "contrasts": {
            'win_anticipation': {'win_event': 1, 'neut_event': -1},
            'loss_anticipation': {'loss_event': 1, 'neut_event': -1},
            'reward_loss_diff': {'win_event': 1, 'loss_event': -1},
            'reward_block': {'win': 1, 'loss': -1}
        }
    },
    "EMOTION": {
        "evs": ['fear', 'neut'],
        "contrasts": {
            'fear_response': {'fear': 1, 'neut': -1}
        }
    },
    "SOCIAL": {
        "evs": ['mental', 'rnd'],
        "contrasts": {
            'theory_of_mind': {'mental': 1, 'rnd': -1}
        }
    },
    "RELATIONAL": {
        "evs": ['relation', 'match'],
        "contrasts": {
            'relational_reasoning': {'relation': 1, 'match': -1}
        }
    },
    "LANGUAGE": {
        "evs": ['story', 'math'],
        "contrasts": {
            'narrative': {'story': 1, 'math': -1}
        }
    },
    "WM": {
        "evs": ['2bk_faces', '2bk_body', '2bk_places', '2bk_tools', '0bk_faces', '0bk_body', '0bk_places', '0bk_tools'],
        "contrasts": {
            'executive_load': {
                '2bk_faces': 0.25, '2bk_body': 0.25, '2bk_places': 0.25, '2bk_tools': 0.25,
                '0bk_faces': -0.25, '0bk_body': -0.25, '0bk_places': -0.25, '0bk_tools': -0.25
            },
            'face_executive': {'2bk_faces': 1, '0bk_faces': -1}
        }
    },
    "MOTOR": {
        "evs": ['lf', 'lh', 'rf', 'rh', 't'],
        "contrasts": {
            'motor_control': {'lf': 0.2, 'lh': 0.2, 'rf': 0.2, 'rh': 0.2, 't': 0.2} # > Baseline (implicit 0)
        }
    }
}

#  GLM WORKER FUNCTION 
def process_subject_task(sub_id, task_name, valid_runs):
    """Fits GLM for a specific subject and task, averages valid runs, saves .npy"""
    runs = valid_runs['Run'].tolist()
    config = TASK_CONFIG[task_name]
    
    # Dictionary to hold the arrays from LR and RL before averaging
    run_contrast_maps = {con_name: [] for con_name in config["contrasts"].keys()}
    
    for run in runs:
        run_dir = os.path.join(BASE_DIR, task_name, sub_id, run)
        ptseries_file = os.path.join(run_dir, f"{sub_id}_{task_name}_{run}.ptseries.nii")
        
        if not os.path.exists(ptseries_file):
            continue

        # LOAD & Z-SCORE TIMESERIES
        try:
            bold_data = nib.load(ptseries_file).get_fdata()
            # Standardise: (X - mean) / std. Add epsilon to avoid divide-by-zero
            bold_data = (bold_data - np.mean(bold_data, axis=0)) / (np.std(bold_data, axis=0) + 1e-10)
        except Exception as e:
            return f"FAILED {sub_id} {task_name} {run}: Data load error {e}"

        n_scans = bold_data.shape[0]
        frame_times = np.arange(n_scans) * TR
        
        # LOAD EVs
        ev_dir = os.path.join(run_dir, "EVs")
        events_list = []
        for ev_name in config["evs"]:
            ev_path = os.path.join(ev_dir, f"{ev_name}.txt")
            if os.path.exists(ev_path) and os.path.getsize(ev_path) > 0:
                try:
                    ev_df = pd.read_csv(ev_path, sep='\t', header=None, names=['onset', 'duration', 'amplitude'])
                    ev_df['trial_type'] = ev_name
                    events_list.append(ev_df)
                except Exception:
                    pass
                    
        if not events_list:
            continue
            
        events_df = pd.concat(events_list, ignore_index=True)
        
        # LOAD MOTION REGRESSORS (6 DOF)
        motion_file = os.path.join(run_dir, "Movement_Regressors.txt")
        try:
            confounds = pd.read_csv(motion_file, delim_whitespace=True, header=None).iloc[:, 0:6]
            confounds.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        except Exception:
            return f"FAILED {sub_id} {task_name} {run}: Motion load error"

        # BUILD DESIGN MATRIX
        design_matrix = make_first_level_design_matrix(
            frame_times, 
            events_df, 
            hrf_model='spm', 
            drift_model='cosine', 
            high_pass=0.01,
            add_regs=confounds
        )
        
        # RUN THE GLM
        labels, estimates = run_glm(bold_data, design_matrix.values)
        cols = design_matrix.columns.tolist()
        
        # COMPUTE CONTRASTS
        for con_name, con_dict in config["contrasts"].items():
            vec = np.zeros(len(cols))
            for cond, weight in con_dict.items():
                if cond in cols:
                    vec[cols.index(cond)] = weight
            
            # Extract Effect Size (Beta)
            contrast_map = compute_contrast(labels, estimates, vec, stat_type='t').effect_size()
            run_contrast_maps[con_name].append(contrast_map)

    # AVERAGE AND SAVE
    saved_count = 0
    for con_name, maps in run_contrast_maps.items():
        if len(maps) > 0:
            avg_map = np.mean(np.array(maps), axis=0) # Simple mean of LR and RL
            out_file = os.path.join(OUT_DIR, f"{sub_id}_{task_name}_{con_name}.npy")
            np.save(out_file, avg_map)
            saved_count += 1
            
    if saved_count > 0:
        return f"SUCCESS: {sub_id} | {task_name} | Runs averaged: {len(run_contrast_maps[list(config['contrasts'].keys())[0]])}"
    return f"FAILED: {sub_id} | {task_name} | No valid runs processed"


# EXECUTION 
if __name__ == "__main__":
    logging.info(" STARTING PHASE 0D: UNIFIED FIRST-LEVEL GLMs ")
    
    df = pd.read_csv(AUDIT_FILE)
    valid_data = df[df['EXCLUDE_RUN'] == False]
    
    # Create unique Subject-Task pairs to process
    sub_tasks = valid_data[['Subject', 'Task']].drop_duplicates().values.tolist()
    
    logging.info(f"Total Subject-Task pairings to model: {len(sub_tasks)}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = []
        for sub_id, task_name in sub_tasks:
            sub_valid_runs = valid_data[(valid_data['Subject'] == sub_id) & (valid_data['Task'] == task_name)]
            futures.append(executor.submit(process_subject_task, str(sub_id), task_name, sub_valid_runs))
        
        # Monitor progress
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if "SUCCESS" in res:
                logging.info(f"[{i}/{len(sub_tasks)}] {res}")
            else:
                logging.error(f"[{i}/{len(sub_tasks)}] {res}")

    logging.info(" PHASE 0D COMPLETE ")