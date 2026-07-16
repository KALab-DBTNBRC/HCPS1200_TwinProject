import os
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime

# CONFIGURATION & PATHS
BASE_DIR      = os.environ.get("PROJECT_ROOT", ".")
TEMPLATE_DIR  = os.path.join(BASE_DIR, 'study_template')
# Demographic/ROI file as requested
CSV           = os.path.join(BASE_DIR, 'twintables', 'network_roi_metrics_FINAL.csv')
OLD_STATS_DIR = os.path.join(BASE_DIR, 'stats', 'cfestats') 
SUBJECTS_FILE = os.path.join(OLD_STATS_DIR, 'subjects.txt')

# Outputs go to the filtered folder
NEW_STATS_DIR = os.path.join(BASE_DIR, 'stats', 'cfestats_ACT_SIFT2_filtered')

DESIGN_MAT    = os.path.join(NEW_STATS_DIR, 'design.mat')
EB_TXT        = os.path.join(NEW_STATS_DIR, 'eb.txt')

os.makedirs(NEW_STATS_DIR, exist_ok=True)

# LOAD DATA & ALIGN SUBJECTS
with open(SUBJECTS_FILE) as f:
    subjects = [int(l.strip().replace('sub-', '').replace('.mif', '')) for l in f]

df = pd.read_csv(CSV)
# Align row order to the imaging data
df = df.set_index('Subject').loc[subjects].reset_index()

# Verify alignment
assert list(df['Subject']) == subjects, "CRITICAL: Subject order mismatch."

# CONSTRUCT 8-COLUMN DESIGN MATRIX (Population Sensitivity)
# Col 1: Intercept
# Col 2: AUD Severity (Variable of Interest)
# Col 3: Age (Demeaned)
# Col 4: Gender
# Col 5: Smoking (Current)
# Col 6: Illicit Drug Use (Count)
# Col 7: Family History (Combined score)
# Col 8: Marijuana Use (Count)
design = pd.DataFrame()
design['Intercept']    = np.ones(len(df))
design['Severity']     = df['Severity'].values
design['Age_Demeaned'] = df['Age_in_Yrs'].values - df['Age_in_Yrs'].mean()
design['Gender']       = df['Gender'].values
design['Smoking']      = df['SSAGA_TB_Still_Smoking'].values
design['Illicit_Use']  = df['SSAGA_Times_Used_Illicits'].values
design['Family_Hist']  = df['FamHist_Combined_DrgAlc'].values
design['Mj_Use']       = df['SSAGA_Mj_Times_Used'].values

design.to_csv(DESIGN_MAT, sep=' ', header=False, index=False, float_format='%.6f')

print(f"[{datetime.now()}] Design Matrix constructed with 8 columns.")
print(f"Note: Race/Ethnicity omitted (Zero variance in this cohort).")

# 4. EXCHANGEABILITY BLOCKS
pair_counts = df['TwinPairID'].value_counts()
singletons  = pair_counts[pair_counts == 1].index
df['EB'] = pd.factorize(df['TwinPairID'])[0] + 1
df.loc[df['TwinPairID'].isin(singletons), 'EB'] = 0
df['EB'].to_csv(EB_TXT, header=False, index=False)

# 5. RUN 70-CORE FIXELCFESTATS WITH DUAL MATRICES
# FDC and FD use the SIFT2-weighted matrix (microstructural calibration)
# FC uses the unweighted geometric matrix (macroscopic morphological)
metrics = {
    'fdc': (os.path.join(BASE_DIR, 'fixel_metrics', 'fdc'),
            os.path.join(TEMPLATE_DIR, 'matrix_ACT_SIFT2')),      
    'fd':  (os.path.join(BASE_DIR, 'fixel_metrics', 'fd'),
            os.path.join(TEMPLATE_DIR, 'matrix_ACT_SIFT2')),      
    'fc':  (os.path.join(BASE_DIR, 'fixel_metrics', 'fc'),
            os.path.join(TEMPLATE_DIR, 'matrix_ACT_unweighted'))  
}

# Updated contrasts for the 8-column matrix
# 0 (Int) | 1 (Sev) | 0 (Age) | 0 (Gen) | 0 (Smk) | 0 (Ill) | 0 (FH) | 0 (Mj)
contrasts = {
    'loss': '0 -1 0 0 0 0 0 0',   
    'gain': '0  1 0 0 0 0 0 0',   
}

for metric_name, (input_dir, matrix_dir) in metrics.items():
    metric_stats_dir = os.path.join(NEW_STATS_DIR, f'{metric_name}stats')
    os.makedirs(metric_stats_dir, exist_ok=True)
    
    for name, c_str in contrasts.items():
        res_dir = os.path.join(metric_stats_dir, f'results_{metric_name}_{name}')
        contrast_file = os.path.join(metric_stats_dir, f'contrast_{name}.mat')
        os.makedirs(res_dir, exist_ok=True)

        with open(contrast_file, 'w') as f:
            f.write(c_str + '\n')

        print(f"\n[{datetime.now()}] Running {metric_name.upper()} {name.upper()}...")
        print(f" > Matrix used: {os.path.basename(matrix_dir)}")
        
        cmd = [
            'fixelcfestats', input_dir, SUBJECTS_FILE, DESIGN_MAT, 
            contrast_file, matrix_dir, res_dir,
            '-errors', 'ee', '-exchange_whole', EB_TXT, 
            '-nthreads', '70', '-nshuffles', '5000', '-force', '-quiet'
        ]
        subprocess.run(cmd, check=True)

print(f"\n[{datetime.now()}] DONE. All results in: {NEW_STATS_DIR}")