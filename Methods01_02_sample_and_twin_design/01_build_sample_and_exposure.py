"""
01_build_sample_and_exposure.py

Corresponds to Methods: "Sample and twin design" and "Degree of alcohol use"
(the first two paragraphs of Methods).

Builds the master analytic sample table from two source files:
  (1) HCP Restricted-Access data -- genotype-confirmed zygosity, family/pair
      structure, and SSAGA alcohol diagnosis fields.
  (2) HCP Open Access data -- sex/gender (used only for same-sex pairing;
      never sourced from the Restricted file).

Neither source file is included in this repository. Both require your own
HCP Data Use Agreement(s) -- Restricted Access for (1), standard Open Access
for (2). See https://www.humanconnectome.org.

This script performs two levels of a single pipeline stage. They are kept in
one file because the second level's exposure variable depends on the exact
subject list produced by the first, and splitting them risked the two
drifting out of sync.
"""

import os
import pandas as pd

RESTRICTED_TABLE = os.environ.get(
    "RESTRICTED_TWIN_TABLE",
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "restricted/HCP_Restricted.csv")
)
OPEN_ACCESS_GENDER_TABLE = os.environ.get(
    "OPEN_ACCESS_GENDER_TABLE",
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "open_access/HCPMainTabGender.csv")
)
IMAGING_COMPLETENESS_TABLE = os.environ.get(
    "IMAGING_COMPLETENESS_TABLE",
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "qc/imaging_completeness.csv")
)
OUTPUT_TABLE = os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables", "master_sample.csv")

EXCLUDE_PAIRS = ["Pair41"]  # incomplete diffusion acquisition; see Methods


# LEVEL 1 -- Sample and twin design
# Restrict to complete, same-sex twin pairs with genotype-confirmed
# zygosity. Zygosity is taken from the genotyped field (ZygosityGT),
# never from self-report (ZygositySR), per the Methods text.

def build_zygosity_and_pairing(restricted_path, gender_path):
    restricted = pd.read_csv(restricted_path)
    gender = pd.read_csv(gender_path)  # Open Access -- Subject, Gender only

    # Genotype-confirmed zygosity only
    gt = restricted[restricted["ZygosityGT"].isin(["MZ", "DZ"])].copy()

    # Join sex from Open Access data (never from the Restricted file)
    gt = gt.merge(gender[["Subject", "Gender"]], on="Subject", how="left")

    # Keep only families that are (a) complete pairs and (b) same-sex
    fam_size = gt.groupby("Family_ID").size()
    complete_fams = fam_size[fam_size == 2].index
    fam_sex_nunique = gt[gt["Family_ID"].isin(complete_fams)].groupby("Family_ID")["Gender"].nunique()
    same_sex_complete_fams = fam_sex_nunique[fam_sex_nunique == 1].index

    sample = gt[gt["Family_ID"].isin(same_sex_complete_fams)].copy()
    sample = sample.rename(columns={"Family_ID": "TwinPairID", "ZygosityGT": "ZygosityGT1"})

    if "Pair41" in sample["TwinPairID"].astype(str).values:
        sample = sample[~sample["TwinPairID"].astype(str).isin(EXCLUDE_PAIRS)]

    return sample[["Subject", "TwinPairID", "ZygosityGT1", "Gender", "Age_in_Yrs"]]


# LEVEL 2 -- Degree of alcohol use
# Ordinal clinical-severity score from SSAGA DSM-IV abuse/dependence
# diagnoses. Coding note: raw SSAGA diagnosis fields use 1=No, 5=Yes,
# not the more common 0/1 convention -- handled explicitly below.
# The two diagnoses are mutually exclusive in this cohort (verified:
# zero subjects meet criteria for both), giving a clean 0/1/2 gradient.

def compute_severity(restricted_path, sample_subjects):
    restricted = pd.read_csv(restricted_path)
    ssaga = restricted[restricted["Subject"].isin(sample_subjects)][
        ["Subject", "SSAGA_Alc_D4_Ab_Dx", "SSAGA_Alc_D4_Dp_Dx"]
    ].copy()

    def severity(row):
        if row["SSAGA_Alc_D4_Dp_Dx"] == 5:
            return 2  # dependence
        if row["SSAGA_Alc_D4_Ab_Dx"] == 5:
            return 1  # abuse
        return 0      # no diagnosis

    ssaga["Severity"] = ssaga.apply(severity, axis=1)
    return ssaga[["Subject", "Severity"]]


def flag_discordance(sample_df):
    pair_range = sample_df.groupby("TwinPairID")["Severity"].agg(lambda s: s.max() - s.min())
    discordant_pairs = pair_range[pair_range > 0].index
    sample_df["Discordant"] = sample_df["TwinPairID"].isin(discordant_pairs)
    return sample_df


if __name__ == "__main__":
    print("Level 1 -- building zygosity- and pairing-restricted sample...")
    sample = build_zygosity_and_pairing(RESTRICTED_TABLE, OPEN_ACCESS_GENDER_TABLE)
    print(f"  {sample['Subject'].nunique()} subjects in {sample['TwinPairID'].nunique()} same-sex, "
          f"genotype-confirmed complete pairs.")

    # NOTE: imaging-completeness intersection happens here in the full
    # pipeline. Per-modality QC gates (diffusion, resting-state, task fMRI)
    # are separate scripts elsewhere in this repository; task-fMRI QC in
    # particular removes additional subjects and is documented in its own
    # folder rather than duplicated here.
    if os.path.exists(IMAGING_COMPLETENESS_TABLE):
        completeness = pd.read_csv(IMAGING_COMPLETENESS_TABLE)
        sample = sample[sample["Subject"].isin(completeness["Subject"])]
        print(f"  After imaging-completeness intersection: {sample['Subject'].nunique()} subjects.")
    else:
        print("  IMAGING_COMPLETENESS_TABLE not found -- skipping intersection "
              "(set the environment variable to enable).")

    print("\nLevel 2 -- computing degree of alcohol use...")
    severity = compute_severity(RESTRICTED_TABLE, sample["Subject"])
    sample = sample.merge(severity, on="Subject", how="left")
    sample = flag_discordance(sample)

    print("  Severity distribution:", sample["Severity"].value_counts().sort_index().to_dict())
    print("  Discordant pairs:", sample[sample["Discordant"]]["TwinPairID"].nunique())

    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    sample.to_csv(OUTPUT_TABLE, index=False)
    print(f"\nMaster sample table written to {OUTPUT_TABLE}")
