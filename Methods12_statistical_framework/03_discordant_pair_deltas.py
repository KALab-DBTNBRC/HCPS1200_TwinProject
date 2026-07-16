import os
import pandas as pd
import numpy as np

# Load ROI metrics
df = pd.read_csv(os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "network_roi_metrics_FINAL.csv"))

# Sort by TwinPairID and Severity to organize twins within the pair
# This ensures that for discordant pairs, Twin 1 is Healthy and Twin 2 is Severe.
df = df.sort_values(['TwinPairID', 'Severity'])

# Create Wide Format (Twin 1 and Twin 2 on one row)
twin1 = df.groupby('TwinPairID').nth(0).reset_index()
twin2 = df.groupby('TwinPairID').nth(1).reset_index()

# Merge twins into a single pair-wise dataframe
pairs_df = pd.merge(twin1, twin2, on='TwinPairID', suffixes=('_T1', '_T2'))

# Calculate Deltas for all FBA metrics and networks
metrics = ['FDC', 'FD', 'FC']
networks = ['Reward', 'Salience', 'DMN', 'Olfactory']

delta_cols = []
for metric in metrics:
    for net in networks:
        col = f'{metric}_{net}'
        delta_col = f'Delta_{col}'
        # Delta = Twin_Severe - Twin_Healthy (for discordant pairs)
        pairs_df[delta_col] = pairs_df[f'{col}_T2'] - pairs_df[f'{col}_T1']
        delta_cols.append(delta_col)

# Select relevant columns
info_cols = ['TwinPairID', 'ZygosityGT1_T1', 'Subject_T1', 'Subject_T2', 'Severity_T1', 'Severity_T2']
final_table = pairs_df[info_cols + delta_cols]

# Save the output
final_table.to_csv(os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "twin_pair_deltas.csv"), index=False)
print(f"Generated deltas for {len(final_table)} twin pairs.")