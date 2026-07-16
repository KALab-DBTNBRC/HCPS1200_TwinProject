import subprocess
import os
import pandas as pd

# --- CONFIGURATION ---
ATLAS_PATH = "/media/khushbu-lab/D220C69420C67F4B/DTIadjfMRI/Atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S3.dlabel.nii"
OUTPUT_KEY = "/media/khushbu-lab/D220C69420C67F4B/DTIadjfMRI/Atlas/atlas_translation_key.csv"

def extract_atlas_labels():
    print("--- Extracting CIFTI Label Table ---")
    
    # Export the label table to a text file
    label_txt = "label_table.txt"
    cmd = ["wb_command", "-cifti-label-export-table", ATLAS_PATH, "1", label_txt]
    subprocess.run(cmd, check=True)
    
    labels = []
    with open(label_txt, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                # Format: Name \n Index Red Green Blue Alpha
                name = line.strip()
                try:
                    stats_line = next(f).split()
                    idx = int(stats_line[0])
                    labels.append({'Index': idx, 'LabelName': name})
                except StopIteration:
                    break
    
    df = pd.DataFrame(labels)
    df.to_csv(OUTPUT_KEY, index=False)
    print(f"Success! Translation key saved to: {OUTPUT_KEY}")
    print(f"Total parcels found: {len(df)}")
    
    os.remove(label_txt)

if __name__ == "__main__":
    extract_atlas_labels()