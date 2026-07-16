import pandas as pd
import os

def build_ultimate_dataset():
    # Load the files
    clean_df = pd.read_csv(os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats/network_roi_metrics.csv"))
    main_df = pd.read_csv(os.environ.get(
        "RESTRICTED_TWIN_TABLE",
        os.path.join(os.environ.get("PROJECT_ROOT", "."), "restricted/HCP_Restricted.csv")
    ))
    
    # Force Subject IDs to string to avoid merge errors
    clean_df['Subject'] = clean_df['Subject'].astype(str)
    main_df['Subject'] = main_df['Subject'].astype(str)
    
    # 2. Define columns
    columns = [
        'Subject',
        # Tier 2: Demographics
        'Race',
        'Ethnicity',
        # Tier 2: Behavioral / Polysubstance
        'SSAGA_ChildhoodConduct',
        'SSAGA_Times_Used_Illicits', 
        'SSAGA_Times_Used_Cocaine',
        'SSAGA_Times_Used_Hallucinogens',
        'SSAGA_Times_Used_Sedatives',
        'SSAGA_Times_Used_Stimulants',
        'SSAGA_Times_Used_Opiates',
        'SSAGA_Mj_Times_Used',
        # Tier 2: Alcohol Specificity
        'Total_Beer_Wine_Cooler_7days',
        'Total_Hard_Liquor_7days',
        'Total_Malt_Liquor_7days',
        'Total_Wine_7days',
        'Total_Other_Alc_7days',
        'Total_Drinks_7days',
        'Total_Any_Tobacco_7days',
        # Tier 3: Family History
        'FamHist_Moth_DrgAlc',
        'FamHist_Fath_DrgAlc'
    ]
    
    # Extract and clean the data
    tier_data = main_df[columns].copy()
    
    # For clinical surveys, missing data (NaN) in these specific fields usually means'No' or '0 times used'. We will fill with 0 to allow statistical regression.
    for col in columns[1:]:
        tier_data[col] = pd.to_numeric(tier_data[col], errors='coerce').fillna(0)
        
    # Create a "Combined Parental History" score for easier modeling
    tier_data['FamHist_Combined_DrgAlc'] = tier_data['FamHist_Moth_DrgAlc'] + tier_data['FamHist_Fath_DrgAlc']
        
    # Merge into the Clean dataset
    final_df = pd.merge(clean_df, tier_data, on='Subject', how='left')
    
    # Save the definitive file
    out_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats/network_roi_metrics_FINAL.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"SUCCESS! Built the definitive dataset with {len(final_df)} subjects.")
    print("Added the following covariates for Sensitivity Analysis:")
    for col in columns[1:] + ['FamHist_Combined_DrgAlc']:
        print(f" - {col}")

if __name__ == "__main__":
    build_ultimate_dataset()