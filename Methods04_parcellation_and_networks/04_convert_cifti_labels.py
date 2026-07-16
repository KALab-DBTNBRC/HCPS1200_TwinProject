import subprocess
import os
import pandas as pd
import logging
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = "/media/khushbu-lab/D220C69420C67F4B/DTIadjfMRI"
ATLAS_DIR = os.path.join(BASE_DIR, "Atlas")
ATLAS_PATH = os.path.join(ATLAS_DIR, "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S3.dlabel.nii")

MASTER_KEY_OUT = os.path.join(ATLAS_DIR, "atlas_master_translation_key.csv")
TARGET_KEY_OUT = os.path.join(ATLAS_DIR, "target_seed_networks.csv")

# --- LOGGING SETUP ---
log_file = os.path.join(ATLAS_DIR, f"cifti_translator_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger()

# --- THE DEFINITIVE TARGET SEED DEFINITIONS ---
# This dictionary captures BOTH the specific requested seeds
# AND the full network sets for the Block 5 Segregation analysis.
TARGET_CIRCUITS = {
    # 1. SPECIFIC SEEDS (Block 4 Causal Circuits)
    'Seed_Olf_Piriform': ['_Pir_ROI'],
    'Seed_Olf_OFC':      ['_13l_ROI', '_OFC_ROI', '_pOFC_ROI'],
    'Seed_Rew_NAcc':     ['NAc-shell', 'NAc-core'],
    'Seed_Rew_vmPFC':    ['_10pp_ROI', '_10d_ROI', '_10v_ROI', '_10r_ROI', '_25_ROI'],
    
    # 2. FULL NETWORKS (Block 5 Segregation Analysis)
    'Network_Reward': [
        '_10pp_ROI', '_10d_ROI', '_10v_ROI', '_10r_ROI', '_11l_ROI', '_13l_ROI', 
        '_OFC_ROI', '_pOFC_ROI', '_25_ROI', '_s32_ROI', '_AVI_ROI', '_AAIC_ROI', 
        '_a24_ROI', '_p24_ROI', 'NAc-shell', 'NAc-core', 'CAU-VA', 'CAU-DA', 
        'CAU-body', 'CAU-tail', 'PUT-VA', 'PUT-DA', 'PUT-VP', 'PUT-DP', 'lAMY', 'mAMY'
    ],
    
    'Network_Salience': [
        '_AVI_ROI', '_AAIC_ROI', '_MI_ROI', '_p24pr_ROI', '_a24pr_ROI', '_24dd_ROI',
        '_IFJa_ROI', '_IFJp_ROI', '_IFSp_ROI', '_FOP1_ROI', '_FOP2_ROI', '_FOP3_ROI',
        'lAMY', 'mAMY', 'THA-DAm'
    ],
    
    'Network_DMN': [
        '_31pd_ROI', '_31a_ROI', '_31pv_ROI', '_v23ab_ROI', '_d23ab_ROI', '_RSC_ROI',
        '_10r_ROI', '_10d_ROI', '_10v_ROI', '_PCV_ROI', '_7m_ROI', '_PGi_ROI', 
        '_PGp_ROI', '_STSda_ROI', '_STSdp_ROI', '_PHA1_ROI', '_PHA2_ROI', '_PHA3_ROI',
        'HIP-head', 'HIP-body', 'HIP-tail'
    ],
    
    'Network_Olfactory': [
        '_Pir_ROI', '_EC_ROI', '_PreS_ROI', '_TGd_ROI', '_TGv_ROI', '_OFC_ROI', 
        '_pOFC_ROI', '_13l_ROI', '_25_ROI', 'lAMY', 'mAMY', 'HIP-head', 'HIP-body', 'HIP-tail'
    ]
}

def extract_atlas_labels():
    logger.info("=== STEP 1: Extracting CIFTI Label Table ===")
    
    label_txt = os.path.join(ATLAS_DIR, "temp_label_table.txt")
    
    # 1. Run Connectome Workbench to dump the label dictionary
    cmd = ["wb_command", "-cifti-label-export-table", ATLAS_PATH, "1", label_txt]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("  -> wb_command execution successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"  -> wb_command failed: {e.stderr}")
        return None

    # 2. Parse the text file
    labels = []
    with open(label_txt, 'r') as f:
        for line in f:
            line = line.strip()
            # Labels start with text, followed by a line with integer/colors
            if not line.startswith('#') and line:
                name = line
                try:
                    stats_line = next(f).split()
                    idx = int(stats_line[0])
                    labels.append({'CIFTI_Index': idx, 'Label_Name': name})
                except StopIteration:
                    break
    
    # 3. Clean up the temp file
    if os.path.exists(label_txt):
        os.remove(label_txt)
        
    df = pd.DataFrame(labels)
    df.to_csv(MASTER_KEY_OUT, index=False)
    logger.info(f"  -> Master Translation Key saved! Total Parcels: {len(df)}")
    
    return df

def map_target_networks(df):
    logger.info("=== STEP 2: Isolating Target Seed Networks ===")
    
    mapped_data = []
    
    # Iterate through our defined dictionary and find the exact CIFTI matches
    for circuit_name, search_strings in TARGET_CIRCUITS.items():
        found_labels = 0
        for search_str in search_strings:
            # Find any label that contains our target string
            matches = df[df['Label_Name'].str.contains(search_str, case=False, na=False)]
            
            for _, row in matches.iterrows():
                mapped_data.append({
                    'Circuit_Node': circuit_name,
                    'Label_Name': row['Label_Name'],
                    'CIFTI_Index': row['CIFTI_Index']
                })
                found_labels += 1
                
        logger.info(f"  -> {circuit_name}: Found {found_labels} matching parcels.")

    target_df = pd.DataFrame(mapped_data)
    target_df.to_csv(TARGET_KEY_OUT, index=False)
    logger.info(f"  -> Target Network Key saved to: {TARGET_KEY_OUT}")
    
def main():
    logger.info("Starting CIFTI Atlas Translation Pipeline...")
    if not os.path.exists(ATLAS_PATH):
        logger.error(f"Atlas not found at: {ATLAS_PATH}")
        return
        
    master_df = extract_atlas_labels()
    
    if master_df is not None:
        map_target_networks(master_df)
        
    logger.info("Pipeline Complete. Check log for details.")

if __name__ == "__main__":
    main()