# 12_energy_heritability.R
# ------------------------------------------------------------------------
# Heritability of the 8 raw network-level control energy phenotypes
# (MIND/SC x Reward/Olfactory/Salience/DMN), split out from the combined
# energy+burden lean script for clarity -- this half is complete and
# runs end-to-end once the recovered control-energy script has produced
# subject_network_energy.csv.
#
# ACE model, phenotype residualisation, and LRT structure copied exactly
# from the recovered nct_q45_heritability.R (B1-matched: profile-
# likelihood CIs, AE via omxSetParameters, scale(residuals(lm())),
# LRT_C = ACE vs AE, LRT_A = AE vs E, FDR within family).
# ------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(OpenMx)
  library(dplyr)
})

CTRL_DIR <- Sys.getenv("PROJECT_ROOT", ".")
ENERGY_DIR <- file.path(CTRL_DIR, "control_energy")
BEH_CSV  <- file.path(Sys.getenv("PROJECT_ROOT", "."), "twintables", "network_roi_metrics_FINAL.csv")
OUT_DIR  <- file.path(CTRL_DIR, "heritability")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ALPHA <- 0.05
NETWORK_ORDER <- c("Reward", "Olfactory", "Salience", "DMN")

mxOption(NULL, "Number of Threads", 1L)
mxOption(NULL, "Default optimizer", "SLSQP")

cat(sprintf("[%s] Control energy heritability (8 phenotypes)\n", format(Sys.time(), "%H:%M:%S")))


# 1. LOAD PHENOTYPES
energy_df <- read.csv(file.path(ENERGY_DIR, "subject_network_energy.csv"), stringsAsFactors = FALSE)
energy_cols <- paste0(rep(c("MIND", "SC"), each = 4), "_E_", rep(NETWORK_ORDER, 2))
stopifnot(all(energy_cols %in% colnames(energy_df)))
pheno <- energy_df[, c("subj_id", energy_cols)]
cat(sprintf("  Phenotypes: %d\n", length(energy_cols)))


# 2. METADATA + TWIN-PAIR STRUCTURE
beh <- read.csv(BEH_CSV, stringsAsFactors = FALSE)
beh$Subject <- as.integer(beh$Subject)
pheno$subj_id <- as.integer(pheno$subj_id)

df <- pheno %>%
  left_join(beh %>% select(Subject, TwinPairID, ZygosityGT1, Age_in_Yrs, Gender),
            by = c("subj_id" = "Subject")) %>%
  mutate(row = row_number() - 1L)

pair_df <- df %>%
  filter(!is.na(TwinPairID)) %>%
  group_by(TwinPairID) %>%
  arrange(subj_id) %>%
  filter(n() == 2) %>%
  summarise(subj_T1 = subj_id[1], subj_T2 = subj_id[2],
            row_T1 = row[1], row_T2 = row[2],
            ZygLabel = ZygosityGT1[1], .groups = "drop") %>%
  mutate(ZygCode = ifelse(grepl("^MZ", ZygLabel, ignore.case = TRUE), "MZ", "DZ"))

n_mz <- sum(pair_df$ZygCode == "MZ"); n_dz <- sum(pair_df$ZygCode == "DZ")
cat(sprintf("  Pairs: %d (MZ=%d DZ=%d)\n", nrow(pair_df), n_mz, n_dz))


# 3. RESIDUALISE + SCALE
resid_scale <- function(y, meta_df) {
  d <- data.frame(y = y, age = meta_df$Age_in_Yrs, sex = meta_df$Gender)
  ok <- complete.cases(d)
  out <- rep(NA_real_, length(y))
  if (sum(ok) < 10) return(out)
  fit <- lm(y ~ age + sex, data = d[ok, ])
  out[ok] <- as.numeric(scale(residuals(fit)))
  out
}


# 4. ACE MODEL -- copied exactly from the recovered nct_q45_heritability.R
fit_ace_phenotype <- function(pheno_vals, pair_df_local) {
  T1 <- pheno_vals[pair_df_local$row_T1 + 1L]
  T2 <- pheno_vals[pair_df_local$row_T2 + 1L]
  zy <- pair_df_local$ZygCode
  keep <- !is.na(T1) & !is.na(T2)
  T1 <- T1[keep]; T2 <- T2[keep]; zy <- zy[keep]
  mz <- zy == "MZ"; dz <- zy == "DZ"
  rMZ <- if (sum(mz) > 2) cor(T1[mz], T2[mz]) else NA_real_
  rDZ <- if (sum(dz) > 2) cor(T1[dz], T2[dz]) else NA_real_
  mzData <- data.frame(T1 = T1[mz], T2 = T2[mz])
  dzData <- data.frame(T1 = T1[dz], T2 = T2[dz])

  null_r <- function(s) list(rMZ = rMZ, rDZ = rDZ,
    A_pct = NA_real_, A_lbound = NA_real_, A_ubound = NA_real_,
    C_pct = NA_real_, C_lbound = NA_real_, C_ubound = NA_real_,
    E_pct = NA_real_, E_lbound = NA_real_, E_ubound = NA_real_,
    LRT_C_p = NA_real_, LRT_A_p = NA_real_, Status = s)

  a <- mxMatrix("Full", 1, 1, TRUE, .6, "a11", name = "a")
  c <- mxMatrix("Full", 1, 1, TRUE, .3, "c11", name = "c")
  e <- mxMatrix("Full", 1, 1, TRUE, .6, "e11", name = "e")
  A <- mxAlgebra(a %*% t(a), name = "A")
  C <- mxAlgebra(c %*% t(c), name = "C")
  E <- mxAlgebra(e %*% t(e), name = "E")
  propA <- mxAlgebra(A / (A+C+E), name = "propA")
  propC <- mxAlgebra(C / (A+C+E), name = "propC")
  propE <- mxAlgebra(E / (A+C+E), name = "propE")
  covMZ <- mxAlgebra(rbind(cbind(A+C+E, A+C), cbind(A+C, A+C+E)), name = "expCovMZ")
  covDZ <- mxAlgebra(rbind(cbind(A+C+E, 0.5*A+C), cbind(0.5*A+C, A+C+E)), name = "expCovDZ")
  meanG <- mxMatrix("Full", 1, 2, FALSE, 0, name = "expMean")

  modelMZ <- mxModel("MZ", mxData(mzData, "raw"), a, c, e, A, C, E, covMZ, meanG,
    mxExpectationNormal("expCovMZ", "expMean", dimnames = c("T1", "T2")), mxFitFunctionML())
  modelDZ <- mxModel("DZ", mxData(dzData, "raw"), a, c, e, A, C, E, covDZ, meanG,
    mxExpectationNormal("expCovDZ", "expMean", dimnames = c("T1", "T2")), mxFitFunctionML())
  ace_model <- mxModel("ACE", a, c, e, A, C, E, propA, propC, propE, modelMZ, modelDZ,
    mxFitFunctionMultigroup(c("MZ", "DZ")), mxCI(c("propA", "propC", "propE")))

  fit_ace <- tryCatch(mxRun(ace_model, intervals = TRUE, silent = TRUE), error = function(e) NULL)
  if (is.null(fit_ace)) return(null_r("FAILED_ACE"))
  code <- fit_ace$output$status$code
  if (!(code %in% c(0L, 1L))) return(null_r(sprintf("CONVERGE_FAIL_%d", code)))

  est_A <- mxEval(propA, fit_ace)[1,1]; est_C <- mxEval(propC, fit_ace)[1,1]; est_E <- mxEval(propE, fit_ace)[1,1]
  cis <- fit_ace$output$confidenceIntervals
  get_ci <- function(nm, bd) { key <- paste0("ACE.", nm, "[1,1]")
    if (!is.null(cis) && key %in% rownames(cis)) cis[key, bd] else NA_real_ }

  fit_ae <- tryCatch({ ae <- mxModel(fit_ace, name = "AE")
    ae <- omxSetParameters(ae, labels = "c11", free = FALSE, values = 0)
    mxRun(ae, silent = TRUE) }, error = function(e) NULL)

  lrt_C_p <- NA_real_
  if (!is.null(fit_ae) && fit_ae$output$status$code %in% c(0L, 1L))
    lrt_C_p <- mxCompare(fit_ace, fit_ae)$p[2]

  lrt_A_p <- NA_real_
  if (!is.null(fit_ae) && fit_ae$output$status$code %in% c(0L, 1L)) {
    fit_e <- tryCatch({ em <- mxModel(fit_ae, name = "E")
      em <- omxSetParameters(em, labels = "a11", free = FALSE, values = 0)
      mxRun(em, silent = TRUE) }, error = function(e) NULL)
    if (!is.null(fit_e) && fit_e$output$status$code %in% c(0L, 1L))
      lrt_A_p <- mxCompare(fit_ae, fit_e)$p[2]
  }

  list(rMZ = round(rMZ, 4), rDZ = round(rDZ, 4),
       A_pct = round(est_A, 4), A_lbound = round(get_ci("propA", "lbound"), 4), A_ubound = round(get_ci("propA", "ubound"), 4),
       C_pct = round(est_C, 4), C_lbound = round(get_ci("propC", "lbound"), 4), C_ubound = round(get_ci("propC", "ubound"), 4),
       E_pct = round(est_E, 4), E_lbound = round(get_ci("propE", "lbound"), 4), E_ubound = round(get_ci("propE", "ubound"), 4),
       LRT_C_p = round(lrt_C_p, 6), LRT_A_p = round(lrt_A_p, 6),
       Status = if (code == 0L) "OK" else "CONVERGE_WARN")
}


# 5. RUN
cat(sprintf("[%s] Running ACE for %d energy phenotypes...\n", format(Sys.time(), "%H:%M:%S"), length(energy_cols)))
results <- list()
for (col in energy_cols) {
  pheno_vals <- resid_scale(as.numeric(df[[col]]), df)
  res <- tryCatch(fit_ace_phenotype(pheno_vals, pair_df),
                   error = function(e) list(Status = paste0("R_ERROR:", conditionMessage(e))))
  res_df <- as.data.frame(lapply(res, function(x) if (length(x) == 0) NA else x[1]))
  res_df$Phenotype <- col
  results[[col]] <- res_df
  cat(sprintf("  %-20s  h2=%s  rMZ=%s  rDZ=%s  %s\n", col,
              ifelse(is.null(res$A_pct) || is.na(res$A_pct), "NA", res$A_pct),
              ifelse(is.null(res$rMZ) || is.na(res$rMZ), "NA", res$rMZ),
              ifelse(is.null(res$rDZ) || is.na(res$rDZ), "NA", res$rDZ),
              res$Status))
}

combined <- do.call(rbind, results)
combined$FDR_q_A <- p.adjust(as.numeric(combined$LRT_A_p), method = "fdr")
combined$Sig_A <- combined$FDR_q_A < ALPHA

out_path <- file.path(OUT_DIR, "nct_energy_heritability.csv")
write.csv(combined, out_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
print(combined)
