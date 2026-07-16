"""
02_mundlak_decomposition_and_equivalence.py

Corresponds to Methods: "Between- versus within-family decomposition"
(Mundlak) and "Equivalence testing and software" (TOST).

Two levels: Mundlak between/within decomposition, then a two-one-sided-
tests (TOST) equivalence test on the within-family term against an
INDEPENDENT bound (the population-level association for that same
metric/network, from Population_Association -- never derived from this
same model's own between-family term, which would be circular).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as st

MASTER_CSV = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv")

# Reduced covariate set for Mundlak decomposition -- deliberately narrower
# than the population-association covariate set (age, sex, smoking only),
# since between/within terms are pair-differenced or pair-meaned and a
# wider covariate set risks degenerate columns within pairs.
REDUCED_COVARIATES = ["Age_in_Yrs", "Gender", "SSAGA_TB_Still_Smoking"]



# LEVEL 1 -- Mundlak between-/within-family decomposition


def mundlak_decompose(df, metric_col, group_col="TwinPairID", severity_col="Severity"):
    """Standardise the outcome, build between-family (pair mean) and
    within-family (deviation from pair mean) severity terms, both
    z-scored, and fit a mixed-effects model with pair as random intercept.
    Returns the fitted model."""
    d = df.copy()
    z_col = f"{metric_col}_z"
    d[z_col] = (d[metric_col] - d[metric_col].mean()) / d[metric_col].std()

    pair_mean = d.groupby(group_col)[severity_col].transform("mean")
    d["BF"] = (pair_mean - pair_mean.mean()) / pair_mean.std()

    within_dev = d[severity_col] - pair_mean
    d["WF"] = (within_dev - within_dev.mean()) / within_dev.std()

    cov_formula = " + ".join(REDUCED_COVARIATES)
    formula = f"{z_col} ~ BF + WF + {cov_formula}"

    fit = smf.mixedlm(formula, d, groups=d[group_col]).fit(reml=False)
    return fit


def run_mundlak_for_metric(df, metric_col):
    fit = mundlak_decompose(df, metric_col)
    return {
        "Metric": metric_col,
        "BF_Beta": fit.params["BF"],
        "BF_P": fit.pvalues["BF"],
        "WF_Beta": fit.params["WF"],
        "WF_SE": fit.bse["WF"],
        "WF_P": fit.pvalues["WF"],
    }


def run_mundlak_zygosity_stratified(df, metric_col, zygosity_col="ZygosityGT1"):
    """Corresponds to Methods: 'Zygosity-stratified decomposition' --
    refit the same Mundlak model within MZ-only and DZ-only strata."""
    results = {}
    for zyg in ["MZ", "DZ"]:
        sub = df[df[zygosity_col] == zyg]
        if len(sub) < 10:
            continue
        try:
            fit = mundlak_decompose(sub, metric_col)
            results[zyg] = {
                "BF_Beta": fit.params["BF"], "BF_P": fit.pvalues["BF"],
                "WF_Beta": fit.params["WF"], "WF_P": fit.pvalues["WF"],
                "N": len(sub),
            }
        except Exception as e:
            results[zyg] = {"error": str(e)}
    return results



# LEVEL 2 -- Equivalence testing (TOST) against an INDEPENDENT bound


def tost_equivalence(wf_beta, wf_se, bound):
    """Two one-sided tests. `bound` MUST be an independent estimate
    (the population-level association for this metric/network from the
    separate population-association model), never this same Mundlak
    model's own between-family coefficient -- using a model's own term
    as its own equivalence yardstick is circular and was an error caught
    and fixed during this project's manuscript audit."""
    t_lower = (wf_beta - (-bound)) / wf_se
    t_upper = (wf_beta - bound) / wf_se
    p_lower = 1 - st.norm.cdf(t_lower)
    p_upper = st.norm.cdf(t_upper)
    tost_p = max(p_lower, p_upper)
    return {
        "Bound": bound,
        "TOST_p_lower": p_lower,
        "TOST_p_upper": p_upper,
        "TOST_p": tost_p,
        "Equivalence_Established": tost_p < 0.05,
    }


if __name__ == "__main__":
    df = pd.read_csv(MASTER_CSV, low_memory=False)

    # Independent population-level bounds -- sourced from the separate
    # population-association model (Population_Association.py /
    # lmemodel_dti_standardized.py), NEVER from this script's own BF term.
    # Example values as verified for Reward and Olfactory (limbic-
    # olfactory) network MD during this project's audit; update per
    # metric/network as needed from your own population-association output.
    INDEPENDENT_BOUNDS = {
        "MD_Reward": 0.169,
        "MD_Olfactory": 0.202,
    }

    all_results = []
    for metric, bound in INDEPENDENT_BOUNDS.items():
        if metric not in df.columns:
            print(f"  > Skipping {metric} (column missing)")
            continue

        pooled = run_mundlak_for_metric(df, metric)
        equiv = tost_equivalence(pooled["WF_Beta"], pooled["WF_SE"], bound)
        strat = run_mundlak_zygosity_stratified(df, metric)

        print(f"\n{metric}:")
        print(f"  Pooled BF beta={pooled['BF_Beta']:.4f} (p={pooled['BF_P']:.4f})  "
              f"WF beta={pooled['WF_Beta']:.4f} (SE={pooled['WF_SE']:.4f}, p={pooled['WF_P']:.4f})")
        print(f"  TOST vs independent bound +/-{bound}: p={equiv['TOST_p']:.4f}  "
              f"Equivalence established: {equiv['Equivalence_Established']}")
        for zyg, res in strat.items():
            if "error" not in res:
                print(f"  {zyg}-only: BF beta={res['BF_Beta']:.4f} (p={res['BF_P']:.4f})  "
                      f"WF beta={res['WF_Beta']:.4f} (p={res['WF_P']:.4f})  N={res['N']}")

        all_results.append({**pooled, **equiv, "Metric": metric})

    out_df = pd.DataFrame(all_results)
    out_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "mundlak_equivalence_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
