"""
11_network_stats.py

Corresponds to Methods: per-network statistics for both fixel-based and conventional tensor metrics -- explicitly parallel, matching the "FBA-versus-DTI contrast in the Results" described in Methods. Two levels, both required.
"""

import os
import subprocess
import pandas as pd



# LEVEL 1 -- per-network fixel-based (FD/FC/FDC) statistics extraction (orig. phase9A2networkstats.py)


def run_clean_extraction():
    project_dir = os.environ.get("PROJECT_ROOT", ".")
    fixel_metrics_dir = f"{project_dir}/fixel_metrics"
    mask_base = f"{project_dir}/final_masks/fixel"
    
    metrics = ["fd", "fc", "fdc"]
    networks = ["DMN", "Olfactory", "Reward", "Salience"]
    
    subjects = sorted([f.replace('.mif', '') for f in os.listdir(f"{fixel_metrics_dir}/fd") if f.endswith('.mif')])
    
    all_data = []

    print(f"--- RE-EXTRACTING CLEAN ROI DATA FOR {len(subjects)} SUBJECTS ---")

    for i, sub in enumerate(subjects):
        sub_row = {'Subject': sub}
        
        for metric in metrics:
            metric_file = f"{fixel_metrics_dir}/{metric}/{sub}.mif"
            
            for net in networks:
                mask_path = f"{mask_base}/{net}_fixel_mask_bin.mif"
                cmd = f"mrstats {metric_file} -mask {mask_path} -ignorezero -output mean"
                result = subprocess.run(cmd.split(), shell=False, capture_output=True, text=True)
                mean_val = result.stdout.strip()
                
                if mean_val:
                    sub_row[f"{metric.upper()}_{net}"] = float(mean_val)
        
        all_data.append(sub_row)
        if (i + 1) % 20 == 0:
            print(f"  > Processed {i + 1}/{len(subjects)} subjects...")

    #Merge with Metadata
    clean_imaging_df = pd.DataFrame(all_data)
    
    old_csv_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "stats/network_stats/roi_extraction/network_roi_metrics_FINAL.csv")
    if os.path.exists(old_csv_path):
        print("\nMerging with clinical metadata...")
        meta_df = pd.read_csv(old_csv_path)

        metadata_cols = ['Subject', 'TwinPairID', 'ZygosityGT1', 'Severity', 'Age_in_Yrs', 'Gender', 'SSAGA_TB_Still_Smoking', 'SSAGA_Alc_D4_Ab_Dx', 'SSAGA_Alc_D4_Dp_Dx', 'SSAGA_Alc_12_Drinks_Per_Day', 'SSAGA_Alc_12_Frq', 'Total_Drinks_7days_x', 'Num_Days_Drank_7days', 'Risky_12Month', 'Risky_Drinks_7days', 'SSAGA_FTND_Score', 'DSM_Anxi_T', 'SSAGA_HSI_Score', 'SSAGA_Alc_Age_1st_Use', 'SSAGA_TB_Still_Smoking', 'SSAGA_TB_Smoking_History']
        meta_only = meta_df[metadata_cols].drop_duplicates()
        
        # Standardize Subject IDs
        clean_imaging_df['Subject'] = clean_imaging_df['Subject'].astype(str).str.replace('sub-', '', regex=False)
        meta_only['Subject'] = meta_only['Subject'].astype(str).str.replace('sub-', '', regex=False)
        
        final_df = pd.merge(meta_only, clean_imaging_df, on='Subject', how='inner')
        
        if len(final_df) == 0:
            print("\n[!] WARNING: Still 0 rows. Check imaging vs metadata IDs.")
        else:
            out_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats/network_roi_metrics.csv")
            final_df.to_csv(out_path, index=False)
            print(f"\nSUCCESS! Clean Master Table saved to: {out_path}")
            print(f"Final Row Count: {len(final_df)}")
            print("First few IDs merged successfully:", final_df['Subject'].head().tolist())
    else:
        print(f"\n[!] Error: Could not find clinical file at {old_csv_path}")

def run_level_1():
    run_clean_extraction()


# LEVEL 2 -- per-network conventional-tensor (FA/MD/RD/AD) statistics extraction (orig. phase9A2networkstatsconventionaldti.py)


def run_dti_extraction():
    project_dir = os.environ.get("PROJECT_ROOT", ".")
    dti_metrics_dir = f"{project_dir}/dti_metrics/warped"
    
    # Using the voxel masks generated from tckmap -> mrthreshold
    mask_base = f"{project_dir}/study_template" 
    
    metrics = ["fa", "md", "rd", "ad"]
    networks = ["DMN", "Olfactory", "Reward", "Salience"]
    
    # Dynamically grab successful subjects from the warped FA directory
    fa_dir = f"{dti_metrics_dir}/fa"
    subjects = sorted([f.replace('sub-', '').replace('_fa_warped.mif', '') 
                       for f in os.listdir(fa_dir) if f.endswith('_warped.mif')])
    
    all_data = []

    print(f"--- EXTRACTING DTI ROI DATA FOR {len(subjects)} SUBJECTS ---")

    for i, sub in enumerate(subjects):
        sub_row = {'Subject': sub}
        
        for metric in metrics:
            # Matches the output format from our Phase 7 HP script
            metric_file = f"{dti_metrics_dir}/{metric}/sub-{sub}_{metric}_warped.mif"
            
            for net in networks:
                mask_path = f"{mask_base}/{net}_voxel_mask.mif"
                
                if not os.path.exists(mask_path):
                    if i == 0: print(f"[!] Warning: Missing mask {mask_path}")
                    continue
                    
                cmd = f"mrstats {metric_file} -mask {mask_path} -ignorezero -output mean"
                result = subprocess.run(cmd.split(), shell=False, capture_output=True, text=True)
                mean_val = result.stdout.strip()
                
                if mean_val:
                    # Store as FA_Olfactory, MD_Reward, etc.
                    sub_row[f"{metric.upper()}_{net}"] = float(mean_val)
        
        all_data.append(sub_row)
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(subjects)} subjects...")

    clean_imaging_df = pd.DataFrame(all_data)

    print("\n--- MERGING WITH MASTER METADATA ---")
    meta_path = f"{project_dir}/twintables/network_roi_metrics_FINAL.csv"
    
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)

        # Exact metadata columns
        metadata_cols = ['Subject', 'TwinPairID', 'ZygosityGT1', 'Severity', 'Age_in_Yrs', 'Gender', 
                         'SSAGA_TB_Still_Smoking', 'SSAGA_Alc_D4_Ab_Dx', 'SSAGA_Alc_D4_Dp_Dx', 
                         'SSAGA_Alc_12_Drinks_Per_Day', 'SSAGA_Alc_12_Frq', 'Total_Drinks_7days_x', 
                         'Num_Days_Drank_7days', 'Risky_12Month', 'Risky_Drinks_7days', 'SSAGA_FTND_Score', 
                         'DSM_Anxi_T', 'SSAGA_HSI_Score', 'SSAGA_Alc_Age_1st_Use', 'SSAGA_TB_Smoking_History']
        
        # Add the specific beverage columns so correlate them directly later
        bev_cols = ['Total_Beer_Wine_Cooler_7days', 'Total_Hard_Liquor_7days', 'Total_Malt_Liquor_7days', 'Total_Wine_7days']
        metadata_cols.extend([col for col in bev_cols if col in meta_df.columns])
        
        meta_only = meta_df[metadata_cols].drop_duplicates()
        
        # Standardize Subject IDs for perfect merging
        clean_imaging_df['Subject'] = clean_imaging_df['Subject'].astype(str).str.replace('sub-', '', regex=False)
        meta_only['Subject'] = meta_only['Subject'].astype(str).str.replace('sub-', '', regex=False)
        
        final_df = pd.merge(meta_only, clean_imaging_df, on='Subject', how='inner')
        
        if len(final_df) == 0:
            print("\n[!] WARNING: Merge resulted in 0 rows. Check imaging vs metadata IDs.")
        else:
            out_path = f"{project_dir}/final_stats/dti_roi_metrics_FINAL.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            final_df.to_csv(out_path, index=False)
            print(f"SUCCESS! Final table with {len(final_df)} subjects saved to:\\n  -> {out_path}")
            
            # Print a quick preview of the new metric columns
            dti_cols = [c for c in final_df.columns if any(m in c for m in ['FA_', 'MD_', 'RD_', 'AD_'])]
            print(f"\nPreview of extracted DTI metrics:\n{final_df[['Subject'] + dti_cols[:4]].head()}")
    else:
        print(f"ERROR: Could not find master table at {meta_path}")

def run_level_2():
    run_dti_extraction()


if __name__ == "__main__":
    run_level_1()  # Level 1
    run_level_2()  # Level 2