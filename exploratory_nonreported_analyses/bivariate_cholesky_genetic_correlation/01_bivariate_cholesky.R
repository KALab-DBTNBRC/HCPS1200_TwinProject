# --- PHASE 5C: BIVARIATE CHOLESKY DECOMPOSITION (ALL 4 NETWORKS) ---
# Decomposes the shared variance between DTI Structure and rsfMRI Function.
library(OpenMx)
library(dplyr)
library(tidyr)

# For parallel processing speed
mxOption(key='Number of Threads', value=parallel::detectCores())
Sys.setenv(OMP_NUM_THREADS=parallel::detectCores())

# 1. PATHS & DATA LOADING
BASE_DIR <- Sys.getenv("PROJECT_ROOT", ".")
MASTER_CSV <- file.path(BASE_DIR, "Tables/MASTER_ROI_METRICS_DTI_FBA.csv")
TIER2_CSV <- file.path(BASE_DIR, "Tables/rsfMRI_Tier2_Decomposed_Metrics.csv")

df_master <- read.csv(MASTER_CSV, stringsAsFactors = FALSE)
df_fmri <- read.csv(TIER2_CSV, stringsAsFactors = FALSE)

df_master$Subject <- gsub("sub-", "", as.character(df_master$Subject))
df_fmri$Subject <- gsub("sub-", "", as.character(df_fmri$Subject))
df_raw <- merge(df_master, df_fmri, by="Subject")

# 2. ROBUST COLUMN FINDER & FACTORING
find_col <- function(pattern, columns) {
  matches <- columns[grep(pattern, columns, ignore.case = TRUE)]
  if(length(matches) > 0) return(matches[1]) else return(NULL)
}

race_col <- find_col("^Race$", names(df_raw))
eth_col  <- find_col("^Ethnicity$", names(df_raw))

if (!is.null(race_col)) df_raw[[race_col]] <- as.factor(df_raw[[race_col]])
if (!is.null(eth_col)) df_raw[[eth_col]] <- as.factor(df_raw[[eth_col]])
df_raw$Gender <- as.factor(df_raw$Gender)

# 3. DEFINE ALL 4 BIVARIATE PAIRS
# Order matters: Variable 1 = Structure (DTI), Variable 2 = Function (rsfMRI)
bivariate_pairs <- list(
  Olfactory = c(V1 = "FD_Olfactory", V2 = "Olfactory_FC_Cortical"),
  Reward    = c(V1 = "FD_Reward",    V2 = "Reward_FC_Cortical"),
  Salience  = c(V1 = "FD_Salience",  V2 = "Salience_FC_Cortical"),
  DMN       = c(V1 = "FD_DMN",       V2 = "DMN_FC_Cortical")
)

potential_covs <- c("Age_in_Yrs", "Gender", race_col, eth_col, 
                    "SSAGA_TB_Still_Smoking", 
                    "SSAGA_Times_Used_Illicits", "FamHist_Combined_DrgAlc", "SSAGA_Mj_Times_Used")

results_list <- list()

cat("\n=======================================================\n")
cat(" STARTING BIVARIATE CHOLESKY: DTI FD × rsfMRI FC (ALL NETWORKS)\n")
cat("=======================================================\n")

# 4. CHOLESKY LOOP PER NETWORK
for (network in names(bivariate_pairs)) {
  cat("\n---> Decomposing Network:", network, "<---\n")
  
  # The crucial double-bracket fix to prevent dplyr crashes
  var1 <- bivariate_pairs[[network]][["V1"]]
  var2 <- bivariate_pairs[[network]][["V2"]]
  
  if(!var1 %in% names(df_raw) || !var2 %in% names(df_raw)) {
    cat("  [!] Missing metrics for", network, "network. Skipping.\n")
    next
  }
  
  # --- STEP A: THE "MANUAL WASH" FOR BOTH VARIABLES ---
  df_washed <- df_raw
  for(var in c(var1, var2)) {
    valid_covs <- c()
    for(cov in potential_covs) {
      if(!cov %in% names(df_raw)) next
      current_data <- df_raw[!is.na(df_raw[[var]]) & !is.na(df_raw[[cov]]), ]
      if(length(unique(na.omit(current_data[[cov]]))) >= 2) valid_covs <- c(valid_covs, cov)
    }
    formula_str <- paste(var, "~", paste(valid_covs, collapse = " + "))
    fit <- lm(as.formula(formula_str), data = df_raw, na.action = na.exclude)
    df_washed[[var]] <- as.numeric(scale(resid(fit)))
  }
  
  # --- STEP B: RESHAPE TO BIVARIATE WIDE FORMAT ---
  temp_df <- df_washed %>% select(TwinPairID, ZygosityGT1, all_of(var1), all_of(var2)) %>% na.omit()
  wide_df <- temp_df %>%
    group_by(TwinPairID) %>%
    mutate(TwinID = row_number()) %>%
    pivot_wider(id_cols = c(TwinPairID, ZygosityGT1), 
                names_from = TwinID, 
                values_from = c(all_of(var1), all_of(var2))) %>%
    filter(!is.na(get(paste0(var1,"_1"))) & !is.na(get(paste0(var1,"_2"))))
  
  # Standardize dimnames for OpenMx: T1_V1, T1_V2, T2_V1, T2_V2
  colnames(wide_df)[colnames(wide_df) == paste0(var1,"_1")] <- "T1_V1"
  colnames(wide_df)[colnames(wide_df) == paste0(var2,"_1")] <- "T1_V2"
  colnames(wide_df)[colnames(wide_df) == paste0(var1,"_2")] <- "T2_V1"
  colnames(wide_df)[colnames(wide_df) == paste0(var2,"_2")] <- "T2_V2"
  
  mzData <- subset(wide_df, ZygosityGT1 == "MZ")
  dzData <- subset(wide_df, ZygosityGT1 == "DZ")
  selVars <- c("T1_V1", "T1_V2", "T2_V1", "T2_V2")
  
  if(nrow(mzData) < 10 || nrow(dzData) < 10) {
    cat("  [!] Insufficient pairs for", network, "\n")
    next
  }

  # --- STEP C: OPENMX CHOLESKY MATRICES ---
  a <- mxMatrix(type="Lower", nrow=2, ncol=2, free=TRUE, values=c(.6, .3, .6), labels=c("a11","a21","a22"), name="a")
  c <- mxMatrix(type="Lower", nrow=2, ncol=2, free=TRUE, values=c(.3, .1, .3), labels=c("c11","c21","c22"), name="c")
  e <- mxMatrix(type="Lower", nrow=2, ncol=2, free=TRUE, values=c(.6, .3, .6), labels=c("e11","e21","e22"), name="e")
  
  A <- mxAlgebra(expression=a %*% t(a), name="A")
  C <- mxAlgebra(expression=c %*% t(c), name="C")
  E <- mxAlgebra(expression=e %*% t(e), name="E")
  V <- mxAlgebra(expression=A + C + E, name="V")
  
  # Genetic and Environmental Correlations
  rA <- mxAlgebra(expression=A[2,1] / sqrt(A[1,1] * A[2,2]), name="rA")
  rC <- mxAlgebra(expression=C[2,1] / sqrt(C[1,1] * C[2,2]), name="rC")
  rE <- mxAlgebra(expression=E[2,1] / sqrt(E[1,1] * E[2,2]), name="rE")
  
  covMZ <- mxAlgebra(expression=rbind(cbind(V, A+C), cbind(A+C, V)), name="expCovMZ")
  covDZ <- mxAlgebra(expression=rbind(cbind(V, 0.5%x%A + C), cbind(0.5%x%A + C, V)), name="expCovDZ")
  meanG <- mxMatrix(type="Full", nrow=1, ncol=4, free=FALSE, values=0, name="expMean")
  
  modelMZ <- mxModel("MZ", mxData(observed=mzData[,selVars], type="raw"), 
                     a, c, e, A, C, E, V, covMZ, meanG,
                     mxExpectationNormal(covariance="expCovMZ", means="expMean", dimnames=selVars), 
                     mxFitFunctionML())
  
  modelDZ <- mxModel("DZ", mxData(observed=dzData[,selVars], type="raw"), 
                     a, c, e, A, C, E, V, covDZ, meanG,
                     mxExpectationNormal(covariance="expCovDZ", means="expMean", dimnames=selVars), 
                     mxFitFunctionML())
  
  chol_model <- mxModel("Bivariate_Cholesky", a, c, e, A, C, E, V, rA, rC, rE,
                        modelMZ, modelDZ,
                        mxFitFunctionMultigroup(c("MZ","DZ")),
                        mxCI(c("rA", "rC", "rE", "a21", "e21")))
  
  # --- STEP D: RUN AND NESTED HYPOTHESIS TESTING ---
  tryCatch({
    fit_chol <- mxRun(chol_model, intervals=TRUE, silent=TRUE)
    
    # Nested Model 1: Drop shared Genetic path
    model_No_Shared_A <- omxSetParameters(fit_chol, labels="a21", free=FALSE, values=0)
    fit_No_Shared_A <- mxRun(model_No_Shared_A, silent=TRUE)
    lrt_A <- mxCompare(fit_chol, fit_No_Shared_A)
    p_shared_A <- lrt_A$p[2]
    
    # Nested Model 2: Drop shared Environmental path
    model_No_Shared_E <- omxSetParameters(fit_chol, labels="e21", free=FALSE, values=0)
    fit_No_Shared_E <- mxRun(model_No_Shared_E, silent=TRUE)
    lrt_E <- mxCompare(fit_chol, fit_No_Shared_E)
    p_shared_E <- lrt_E$p[2]
    
    # Extract Estimates
    est_rA <- mxEval(rA, fit_chol)[1,1]
    est_rE <- mxEval(rE, fit_chol)[1,1]
    
    cis <- fit_chol$output$confidenceIntervals
    ci_rA_L <- ifelse(!is.null(cis), cis["Bivariate_Cholesky.rA[1,1]", "lbound"], NA)
    ci_rA_U <- ifelse(!is.null(cis), cis["Bivariate_Cholesky.rA[1,1]", "ubound"], NA)
    ci_rE_L <- ifelse(!is.null(cis), cis["Bivariate_Cholesky.rE[1,1]", "lbound"], NA)
    ci_rE_U <- ifelse(!is.null(cis), cis["Bivariate_Cholesky.rE[1,1]", "ubound"], NA)
    
    results_list[[network]] <- data.frame(
      Network = network,
      Structural_Metric = var1,
      Functional_Metric = var2,
      Genetic_Corr_rA = est_rA,
      rA_95CI_Lower = ci_rA_L,
      rA_95CI_Upper = ci_rA_U,
      P_Value_Shared_A = p_shared_A,
      Environ_Corr_rE = est_rE,
      rE_95CI_Lower = ci_rE_L,
      rE_95CI_Upper = ci_rE_U,
      P_Value_Shared_E = p_shared_E
    )
    
    cat("  [OK] Model Converged.\n")
    cat(sprintf("       rA = %+.3f (p = %.4f)\n", est_rA, p_shared_A))
    cat(sprintf("       rE = %+.3f (p = %.4f)\n", est_rE, p_shared_E))
    
  }, error = function(err) {
    cat("  [!] Error:", conditionMessage(err), "\n")
  })
}

# 5. FINAL EXPORT
final_table <- do.call(rbind, results_list)
if(!is.null(final_table)) {
  # Add FDR correction for the LRT p-values across all 4 networks
  final_table$FDR_Shared_A <- p.adjust(final_table$P_Value_Shared_A, method="fdr")
  final_table$FDR_Shared_E <- p.adjust(final_table$P_Value_Shared_E, method="fdr")
  
  out_path <- file.path(BASE_DIR, "Tables/Phase5C_Bivariate_Cholesky_Results.csv")
  write.csv(final_table, out_path, row.names=FALSE)
  
  cat("\n===================================================================\n")
  cat(" PHASE 5C: FULL BIVARIATE CHOLESKY ARCHITECTURE RESULTS\n")
  cat("===================================================================\n")
  print(final_table %>% select(Network, Genetic_Corr_rA, P_Value_Shared_A, Environ_Corr_rE, P_Value_Shared_E))
  cat(sprintf("\nResults saved to: %s\n", out_path))
} else {
  cat("\n[!] All Bivariate models failed.\n")
}