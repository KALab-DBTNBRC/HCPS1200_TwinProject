<div align="center">

# What Alcohol-Related Brain Differences Are Made Of

### A Multimodal Twin Study of Structure, Function, and Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-submission--ready-brightgreen)
![Data](https://img.shields.io/badge/data-HCP--S1200-blue)
![N](https://img.shields.io/badge/N-238%20(119%20twin%20pairs)-lightgrey)

**Ritam Kanti Roy¹ · Rahul Kothekar¹ · Tanya Bassi¹ · Andrea Jebel¹ · Shefali Chaudhary² · Siddharth Sarkar³ · Yatan Pal Singh Balhara³ · Khushbu Agarwal¹\***

¹ Department of Cognitive and Computational Neuroscience, National Brain Research Centre, Manesar, India
² Department of Psychiatry, Yale University School of Medicine, USA
³ National Drug Dependence Treatment Centre, AIIMS New Delhi, India

*Correspondence: khushbu.agarwal@nbrc.ac.in*

</div>

---

## The question

Heavier alcohol use is consistently associated with differences in brain structure and function — differences almost universally read as *consequences* of drinking. But that reading rests on cross-sectional comparisons between unrelated people, a design that cannot distinguish a within-person effect of exposure from shared genetic or familial liability that happens to elevate both drinking and the brain phenotype.

We put that reading to the test in a twin sample, using every genetically informed tool the twin design offers — heritability, between/within decomposition, discordant-pair contrasts, formal power and equivalence testing — applied **uniformly across every modality**, not just wherever it happened to be convenient.

## What we found

| | |
|---|---|
| 🧬 **Heritable architecture** | Structural, functional, and control properties of the addiction-relevant circuit were strongly heritable throughout — the positive control that makes every subsequent null informative rather than merely insensitive. |
| 🩸 **A signal, but not the one you'd think** | A conventional model recovers the textbook alcohol–diffusivity association. It doesn't survive scrutiny: absent from fibre-specific metrics, excluded as free water by two independent estimators, unmoved by every measured familial confound tested. |
| ⚖️ **Power, taken seriously** | Where the design had adequate power (reward network), the within-person effect is not just non-significant — it's formally equivalent to zero. Where it didn't (limbic-olfactory), we say so plainly rather than call an underpowered null a discovery. |
| 🌊 **Dynamics that are real, but not alcohol's** | Dynamic functional connectivity is genuine (survives a strict phase-randomized surrogate null) and heritable — yet its association with drinking could not be attributed to exposure either. |

**The throughline:** a genetically informed, multimodal design doesn't hand you a cause. It disciplines what an alcohol–brain association is allowed to mean — and in this cohort, at this exposure level, it means less than the cross-sectional literature usually claims.

---

## Repository structure

Folders are numbered to match the paper's own Methods sections — if you're looking for the code behind a specific analysis, the section number is the fastest way in. Two folders intentionally span a pair of adjacent Methods subsections (`01_02` and `05_06`) because the second subsection in each pair is fully covered inside the first folder's own scripts, not because anything is missing.

```
.
├── Methods01_02_sample_and_twin_design/       Sample construction, exposure coding, discordant-pair
│                                               identification, covariate definitions
├── Methods04_parcellation_and_networks/       Atlas warping, network mask construction (4 a priori networks)
├── Methods05_06_diffusion_fba_pipeline/       Fixel-based analysis (FD/FC/FDC) + conventional DTI tensor
├── Methods07_freewater_and_isotropic_modeling/ NODDI/ISOVF, CSF proxy, the decisive free-water attenuation test
├── Methods08_resting_state_static_connectivity/ Static functional connectivity extraction
├── Methods09_dynamic_functional_connectivity/ Sliding-window dynamic fluidity (surrogate-tested)
├── Methods10_task_fmri/                       First-level GLM, motion QC audits
├── Methods11_structural_connectivity_and_nct/ Structural connectomes, MIND, controllability,
│                                               heritability, AUD associations, control energy & burden
├── Methods12_statistical_framework/           Population association, Mundlak decomposition,
│   │                                           TOST equivalence testing, heritability (ACE/Falconer)
│   ├── dynfc_robustness_battery/              Zygosity-split Mundlak, surrogate/excess reanalysis,
│   │                                           motion/leverage/permutation checks, canonical motion-
│   │                                           controlled result
│   └── heritability_ace_R/                    ACE heritability models (DTI/FBA, sex-stratified, olfactory)
└── exploratory_nonreported_analyses/          Real analyses, run and evaluated, explicitly NOT
    │                                           part of the reported findings — see below
    ├── abandoned_chronnectome_approach/
    ├── beverage_specificity_dz_control/
    ├── bivariate_cholesky_genetic_correlation/
    ├── cfe_whole_tractogram_stats/
    └── failed_bitensor_freewater/
```

### On the exploratory folder — and why it exists

A few analyses in this project were run in full, evaluated honestly, and didn't make it into the paper: a bi-tensor free-water approach that failed its own validation gate before NODDI/ISOVF replaced it; an early dynamic-connectivity method superseded by a simpler, more robust one; whole-tractogram statistics that were never the basis of any reported claim; a genetic-correlation model reported only as an exploratory lead; a DZ-control analysis for a beverage-specificity finding that stayed exploratory throughout.

None of it is deleted. This paper's own argument is that a null result, honestly reported, is worth more than a convenient positive one — the same standard applies to the code, not just the prose.

## The a priori networks

| Network | Colour | Role |
|---|---|---|
| Reward | amber `#E69F00` | Incentive valuation |
| Salience | sky blue `#56B4E9` | Interoceptive/attentional switching |
| Default Mode | teal `#009E73` | Internally directed cognition |
| Limbic-Olfactory | vermilion `#D55E00` | Chemosensory-limbic valuation |

(Okabe-Ito colour-blind-safe palette, locked across every figure in the paper.)

## Reproducibility

- Every script reads paths from a `PROJECT_ROOT` environment variable (plus `RESTRICTED_TWIN_TABLE` / `RESTRICTED_DERIVED_TWIN_TABLE` for HCP Restricted-Access fields). Nothing is hardcoded to a specific server.
- **No data is committed here.** HCP imaging and phenotypic data must be obtained directly from [ConnectomeDB](https://db.humanconnectome.org) under the appropriate Open or Restricted Access Data Use Terms.
- Covariate conventions are modality-specific and stated in each script's header — DTI/tensor metrics use a 9-covariate set including Race/Ethnicity with target and Severity both standardized; rsfMRI static metrics use a 6-covariate set without Race/Ethnicity; the Mundlak decomposition uses a reduced 3-covariate set. These are not interchangeable — conflating them was a real error caught during this project's own internal audit, and the scripts are written so that mistake is harder to repeat silently.
- R scripts (heritability/ACE models) have not been independently executed in the environment used to assemble this repository — they are real, unmodified files, but verify locally before relying on their output.
- A handful of scripts are faithful reconstructions of interactive analysis rather than recovered standalone files. Each says so in its own header, states what it was validated against, and — where a discrepancy remains open — says that too, rather than presenting an unresolved number as settled.

## Citation

If you use this code, please cite the paper (details to be finalized on publication):

> Roy, R.K., Kothekar, R., Bassi, T., Jebel, A., Chaudhary, S., Sarkar, S., Balhara, Y.P.S., Agarwal, K. *What alcohol-related brain differences are made of: a multimodal twin study of structure, function and dynamics.* (2026, in preparation).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 KALab-DBTNBRC.

---

<div align="center">

*Human Connectome Project data used under the HCP Open and Restricted Access Data Use Terms.*
*Restricted phenotypes require an executed HCP Restricted Access agreement and are not redistributed here.*

</div>
