# --- PHASE 3: OLFACTORY UNIVARIATE ACE (Odor_AgeAdj) ---
library(OpenMx)
library(dplyr)
library(tidyr)

# PATHS & LOGGING SETUP
BASE_DIR <- Sys.getenv("PROJECT_ROOT", ".")
LOG_DIR <- file.path(BASE_DIR, "logs")
TABLES_DIR <- file.path(BASE_DIR, "tables")
MASTER_CSV <- file.path(TABLES_DIR, "MASTER_ROI_METRICS_DTI_FBA.csv")
BEHAVIOR_CSV <- file.path(TABLES_DIR, "Global_Behavioral_Averaged.csv")
OUTPUT_CSV <- file.path(TABLES_DIR, "Phase3_Olfactory_ACE_Results.csv")
LOG_FILE <- file.path(LOG_DIR, "Phase3_Olfactory_ACE.log")

if(!dir.exists(LOG_DIR)) dir.create(LOG_DIR, recursive = TRUE)

log_info <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  formatted_msg <- sprintf("[%s] INFO: %s", timestamp, msg)
  cat(formatted_msg, "\n")
  cat(formatted_msg, "\n", file = LOG_FILE, append = TRUE)
}

log_info("--- STARTING PHASE 3: OLFACTORY ACE WITH CONFIDENCE INTERVALS ---")

# DATA LOADING & MERGING
df_master <- read.csv(MASTER_CSV, stringsAsFactors = FALSE)
df_behav <- read.csv(BEHAVIOR_CSV, stringsAsFactors = FALSE)

df_master$Subject <- gsub("sub-", "", as.character(df_master$Subject))
df_behav$Subject <- gsub("sub-", "", as.character(df_behav$Subject))
df <- merge(df_master, df_behav, by="Subject")
log_info(sprintf("Merge complete. N = %d subjects.", nrow(df)))

# COLUMN IDENTIFICATION
zyg_matches <- names(df)[grep("ZYG", names(df), ignore.case = TRUE)]
ZYG_COL <- zyg_matches[1]

pair_matches <- names(df)[grep("PAIR|FAMILY|FAM_ID|TWINID", names(df), ignore.case = TRUE)]
pair_matches <- pair_matches[!pair_matches %in% c("Subject", ZYG_COL)]
PAIR_COL <- pair_matches[1]

log_info(sprintf("Using Zygosity: %s | Pair ID: %s", ZYG_COL, PAIR_COL))

# DATA WASHING
target_vars <- c("Odor_AgeAdj")
potential_covs <- c("Gender", "SSAGA_TB_Still_Smoking", "SSAGA_Times_Used_Illicits", "FamHist_Combined_DrgAlc")

df_washed <- df
for(var in target_vars) {
  if(!var %in% names(df)) next
  valid_covs <- Filter(function(c) c %in% names(df) && length(unique(df[[c]])) >= 2, potential_covs)
  formula_str <- paste(var, "~", paste(valid_covs, collapse = " + "))
  fit_wash <- lm(as.formula(formula_str), data = df, na.action = na.exclude)
  df_washed[[var]] <- as.numeric(scale(resid(fit_wash)))
}

# ACE MODEL EXECUTION
results_list <- list()
for (var in target_vars) {
    # Prepare Wide Data
    temp_df <- df_washed %>% select(all_of(c(PAIR_COL, ZYG_COL, var))) %>% na.omit()
    wide_df <- temp_df %>%
      group_by(!!sym(PAIR_COL)) %>%
      mutate(TwinNum = row_number()) %>%
      pivot_wider(id_cols = c(!!sym(PAIR_COL), !!sym(ZYG_COL)), 
                  names_from = TwinNum, values_from = all_of(var)) %>%
      rename(T1 = `1`, T2 = `2`) %>% filter(!is.na(T1) & !is.na(T2))
    
    mz_indicator <- unique(wide_df[[ZYG_COL]])[grep("MZ|1", unique(wide_df[[ZYG_COL]]), ignore.case=TRUE)][1]
    mzData <- subset(wide_df, wide_df[[ZYG_COL]] == mz_indicator, select = c("T1", "T2"))
    dzData <- subset(wide_df, wide_df[[ZYG_COL]] != mz_indicator, select = c("T1", "T2"))
    
    # Model Matrices
    a <- mxMatrix("Full", 1, 1, TRUE, .6, label="a11", name="a")
    c <- mxMatrix("Full", 1, 1, TRUE, .3, label="c11", name="c")
    e <- mxMatrix("Full", 1, 1, TRUE, .6, label="e11", name="e")
    A <- mxAlgebra(a %*% t(a), name="A"); C <- mxAlgebra(c %*% t(c), name="C"); E <- mxAlgebra(e %*% t(e), name="E")
    pA <- mxAlgebra(A/(A+C+E), name="pA"); pC <- mxAlgebra(C/(A+C+E), name="pC"); pE <- mxAlgebra(E/(A+C+E), name="pE")
    
    covMZ <- mxAlgebra(rbind(cbind(A+C+E, A+C), cbind(A+C, A+C+E)), name="covMZ")
    covDZ <- mxAlgebra(rbind(cbind(A+C+E, 0.5*A+C), cbind(0.5*A+C, A+C+E)), name="covDZ")
    meanG <- mxMatrix("Full", 1, 2, FALSE, 0, name="expMean")

    modMZ <- mxModel("MZ", mxData(mzData, "raw"), a,c,e,A,C,E,covMZ,meanG, mxExpectationNormal("covMZ","expMean",c("T1","T2")), mxFitFunctionML())
    modDZ <- mxModel("DZ", mxData(dzData, "raw"), a,c,e,A,C,E,covDZ,meanG, mxExpectationNormal("covDZ","expMean",c("T1","T2")), mxFitFunctionML())
    
    # Parent model with Likelihood-Based CIs
    model <- mxModel("ACE", a,c,e,A,C,E,pA,pC,pE, modMZ, modDZ, mxFitFunctionMultigroup(c("MZ","DZ")), 
                     mxCI(c("pA","pC","pE")))

    try({
      fit <- mxRun(model, intervals=TRUE, silent=TRUE)
      
      # Extract point estimates (%)
      est_A <- mxEval(pA, fit)[1,1] * 100
      est_C <- mxEval(pC, fit)[1,1] * 100
      est_E <- mxEval(pE, fit)[1,1] * 100
      
      # Extract Confidence Intervals from the output table
      cis <- fit$output$confidenceIntervals
      
      res <- data.frame(
        Metric = var,
        A_pct = est_A, 
        A_lbound = cis["ACE.pA[1,1]", "lbound"] * 100,
        A_ubound = cis["ACE.pA[1,1]", "ubound"] * 100,
        
        C_pct = est_C,
        C_lbound = cis["ACE.pC[1,1]", "lbound"] * 100,
        C_ubound = cis["ACE.pC[1,1]", "ubound"] * 100,
        
        E_pct = est_E,
        E_lbound = cis["ACE.pE[1,1]", "lbound"] * 100,
        E_ubound = cis["ACE.pE[1,1]", "ubound"] * 100
      )
      
      results_list[[var]] <- res
      log_info(sprintf("ACE Results for %s:", var))
      log_info(sprintf("  A: %0.1f%% [%0.1f - %0.1f]", res$A_pct, res$A_lbound, res$A_ubound))
      log_info(sprintf("  C: %0.1f%% [%0.1f - %0.1f]", res$C_pct, res$C_lbound, res$C_ubound))
      log_info(sprintf("  E: %0.1f%% [%0.1f - %0.1f]", res$E_pct, res$E_lbound, res$E_ubound))
    })
}

# SAVE
final_table <- do.call(rbind, results_list)
if(!is.null(final_table)) {
  write.csv(final_table, OUTPUT_CSV, row.names=FALSE)
  log_info(sprintf("Results saved to: %s", OUTPUT_CSV))
}
log_info("--- PHASE 3 COMPLETE ---")