import os
import pandas as pd

#  File Paths 
MASTER_FILE = os.path.join(os.environ.get("PROJECT_ROOT", "."), "tables/MASTER_ROI_METRICS_DTI_FBA.csv")
AUDIT_FILE = os.path.join(os.environ.get("PROJECT_ROOT", "."), "tables/QC_Audit_Report.csv")

#  EXTRACT DISCORDANT PAIRS FROM MASTER TABLE 
print("Extracting Discordant Pairs from Master Table...")
master_df = pd.read_csv(MASTER_FILE)

# Ensure IDs are strings for reliable matching
master_df['Subject'] = master_df['Subject'].astype(str)
master_df['TwinPairID'] = master_df['TwinPairID'].astype(str)

def get_discordant_pairs(df, zygosity):
    """Filters master table for pairs with 1 healthy (0) and 1 affected (>0) twin."""
    zyg_df = df[df['ZygosityGT1'] == zygosity]
    discordant_pairs = []
    
    # Group by TwinPairID to evaluate the pair as a unit
    grouped = zyg_df.groupby('TwinPairID')
    
    for pair_id, group in grouped:
        if len(group) == 2: # Ensure both twins exist in the master table
            subs = group['Subject'].values
            sevs = group['Severity'].values
            
            # Discordance logic: (TwinA=0 AND TwinB>0) OR (TwinA>0 AND TwinB=0)
            if (sevs[0] == 0 and sevs[1] > 0) or (sevs[0] > 0 and sevs[1] == 0):
                discordant_pairs.append((subs[0], subs[1]))
                
    return discordant_pairs

mz_pairs = get_discordant_pairs(master_df, 'MZ')
dz_pairs = get_discordant_pairs(master_df, 'DZ')

print(f"Master Table Audit: Found {len(mz_pairs)} MZ and {len(dz_pairs)} DZ Discordant Pairs.\n")


#  CHECK SURVIVAL AGAINST MOTION AUDIT ACROSS ALL TASKS 
audit_df = pd.read_csv(AUDIT_FILE)
audit_df['Subject'] = audit_df['Subject'].astype(str)

# Dynamically identify all tasks present in the audit
tasks = audit_df['Task'].unique()

def check_pair_survival(task, pairs):
    """Check if both twins in a pair have at least one valid run for a specific task."""
    surviving_count = 0
    lost_pairs = []
    task_df = audit_df[audit_df['Task'] == task]
    
    for t1, t2 in pairs:
        # A subject survives if they have at least one run (LR or RL) that is NOT excluded
        t1_valid = not task_df[(task_df['Subject'] == t1) & (task_df['EXCLUDE_RUN'] == False)].empty
        t2_valid = not task_df[(task_df['Subject'] == t2) & (task_df['EXCLUDE_RUN'] == False)].empty
        
        if t1_valid and t2_valid:
            surviving_count += 1
        else:
            lost_pairs.append((t1, t2))
            
    return surviving_count, lost_pairs

#  GENERATE SUMMARY REPORT 
print(f"{'TASK':<15} | {'MZ SURVIVAL':<15} | {'DZ SURVIVAL':<15}")
print("-" * 50)

for task in sorted(tasks):
    mz_survive, _ = check_pair_survival(task, mz_pairs)
    dz_survive, _ = check_pair_survival(task, dz_pairs)
    
    # Flag tasks that hit the exclusion threshold (>2 pairs lost)
    mz_flag = "!" if (len(mz_pairs) - mz_survive) > 2 else " "
    
    print(f"{task:<15} | {mz_survive:>2}/{len(mz_pairs)} {mz_flag}       | {dz_survive:>2}/{len(dz_pairs)}")

print("\n(!) Indicates MZ survival dropped by > 2 pairs (Note statistical power reduction).")