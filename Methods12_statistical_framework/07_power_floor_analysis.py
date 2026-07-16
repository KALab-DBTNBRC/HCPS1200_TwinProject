import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as st

CSV_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv")
OUT_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "power_floor_MDE.csv")

NETWORKS = ["Reward", "Olfactory"]
METRICS = ["MD", "RD", "AD"]
COVARIATES = ("Age_in_Yrs + Gender + SSAGA_TB_Still_Smoking + C(Race) + C(Ethnicity) + "
              "SSAGA_Times_Used_Illicits + SSAGA_Mj_Times_Used + FamHist_Combined_DrgAlc")


def zscore(s):
    return (s - s.mean()) / s.std()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for net in NETWORKS:
        for tm in METRICS:
            col = f"{tm}_{net}"
            df[f"{col}_z"] = zscore(df[col])

    df["Severity_z"] = zscore(df["Severity"])
    pairmean = df.groupby("TwinPairID")["Severity_z"].transform("mean")
    df["Severity_BF"] = pairmean
    df["Severity_WF"] = df["Severity_z"] - pairmean

    mz = df[df["ZygosityGT1"] == "MZ"].copy()
    mz_pairmean = mz.groupby("TwinPairID")["Severity_z"].transform("mean")
    mz["Severity_BF"] = mz_pairmean
    mz["Severity_WF"] = mz["Severity_z"] - mz_pairmean

    z_crit = st.norm.ppf(0.975) + st.norm.ppf(0.80)
    print(f"z_alpha/2 + z_power = {z_crit:.4f}\n")

    results = []
    for net in NETWORKS:
        for tm in METRICS:
            col = f"{tm}_{net}_z"
            fit = smf.mixedlm(f"{col} ~ Severity_BF + Severity_WF + {COVARIATES}",
                               mz, groups=mz["TwinPairID"]).fit(reml=False)
            se_wf = fit.bse["Severity_WF"]
            mde = z_crit * se_wf
            results.append({"Metric": tm, "Network": net, "SE_WF": se_wf, "MDE_80pct_power": mde})
            print(f"{tm}_{net}: SE_WF={se_wf:.4f} -> MDE at 80% power = {mde:.4f}")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print("\nNOTE: this does not exactly reproduce the manuscript's stated 0.13/0.15 --")
    print("see module docstring for the honest discrepancy and what's been ruled out.")
