import os
import nibabel as nib
import numpy as np
import pandas as pd

# 1. Paths
atlas_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "reference_atlas/Glasser_Tian_Composite_Template.nii.gz")
label_txt_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "reference_atlas/my_atlas_labels.txt")

# 2. Extract unique integers from the composite NIfTI
print("Scanning composite atlas volume...")
img = nib.load(atlas_path)
data = np.round(img.get_fdata()).astype(int) # Ensure strict integers
present_indices = np.unique(data)

# Remove background (0)
present_indices = present_indices[present_indices > 0]
print(f"Found {len(present_indices)} unique ROIs in the volume.")

# 3. Load your text label file
# Adjust 'sep' or column names based on how your text file is delimited
df_labels = pd.read_csv(label_txt_path, sep='\s+', header=None, names=['Index', 'Name', 'R', 'G', 'B', 'A'])

# 4. Map the volume indices to the text names
mapped_labels = df_labels[df_labels['Index'].isin(present_indices)]

# Sort them to ensure they are in the exact structural order of your matrix (1 to 410)
ordered_labels = mapped_labels.sort_values(by='Index')

# 5. Export the validated list
output_list = os.path.join(os.environ.get("PROJECT_ROOT", "."), "reference_atlas/Validated_Composite_Labels.csv")
ordered_labels[['Index', 'Name']].to_csv(output_list, index=False)

print(f"Label order extracted and saved to: {output_list}")

# Print the first few and last few to verify boundaries
print("\nFirst 5 ROIs (Cortical):")
print(ordered_labels.head(5).to_string(index=False))
print("\nLast 5 ROIs (Subcortical):")
print(ordered_labels.tail(5).to_string(index=False))