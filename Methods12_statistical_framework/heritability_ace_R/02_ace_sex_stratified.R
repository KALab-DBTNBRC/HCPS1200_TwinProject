# --- SEX-STRATIFIED ACE HERITABILITY: OPENMX MODEL ---
library(OpenMx)
library(dplyr)
library(tidyr)

# Load the Definitive Dataset
data_path <- file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "network_roi_metrics_FINAL.csv")
if(!file.exists(data_path)) stop("Data file not found at the specified path.")

df_raw <- read.csv(data_path, stringsAsFactors = FALSE)

# ROBUST COLUMN FINDER & FACTORING
find_col <- function(pattern, columns) {
  matches <- columns[grep(pattern, columns, ignore.case = TRUE)]
  if(length(matches) > 0) return(matches[1]) else return(NULL)
}

race_col <- find_col("^Race$", names(df_raw))
eth_col  <- find_col("^Ethnicity$", names(df_raw))

# Convert necessary columns to factors for LM residualization
df_raw[[race_col]] <- as.factor(df_raw[[race_col]])
df_raw[[eth_col]]  <- as.factor(df_raw[[eth_col]])

# DEFINE VARIABLES
metrics <- c("FDC", "FD", "FC")
networks <- c("Reward", "Salience", "DMN", "Olfactory")
target_vars <- c()
for(m in metrics) { 
  for(n in networks) { 
    target_vars <- c(target_vars, paste0(m, "_", n)) 
  } 
}

# Potential Covariates (Note: Gender is removed because we are stratifying by it)
potential_covs <- c("Age_in_Yrs", race_col, eth_col, 
                    "SSAGA_TB_Still_Smoking", 
                    "SSAGA_Times_Used_Illicits", "FamHist_Combined_DrgAlc", "SSAGA_Mj_Times_Used")

# MASTER FUNCTION FOR STRATIFIED RUNS
run_stratified_ace <- function(df_input, gender_label) {
  cat(paste0("\n==========================================\n"))
  cat(paste0("STARTING ANALYSIS FOR: ", gender_label, " (N=", nrow(df_input), ")\n"))
  cat(paste0("==========================================\n"))
  
  # --- STEP A: THE "MANUAL WASH" WITHIN GROUP ---
  df_washed <- df_input
  for(var in target_vars) {
    if(!var %in% names(df_input)) next
    
    # Check for valid levels in covariates within THIS gender group
    valid_covs <- c()
    for(cov in potential_covs) {
      if(length(unique(na.omit(df_input[[cov]]))) >= 2) {
        valid_covs <- c(valid_covs, cov)
      }
    }
    
    # Residualize and scale
    formula_str <- paste(var, "~", paste(valid_covs, collapse = " + "))
    fit <- lm(as.formula(formula_str), data = df_input, na.action = na.exclude)
    df_washed[[var]] <- as.numeric(scale(resid(fit)))
  }
  
  # --- STEP B: OPENMX LOOP ---
  results_list <- list()
  
  for (var in target_vars) {
    if(!var %in% names(df_washed)) next
    cat("Fitting ACE for:", var, "\n")
    
    # Wide Format Reshape
    temp_df <- df_washed %>% select(TwinPairID, ZygosityGT1, all_of(var)) %>% na.omit()
    wide_df <- temp_df %>%
      group_by(TwinPairID) %>%
      mutate(TwinID = row_number()) %>%
      pivot_wider(id_cols = c(TwinPairID, ZygosityGT1), names_from = TwinID, values_from = all_of(var)) %>%
      rename(T1 = `1`, T2 = `2`) %>% filter(!is.na(T1) & !is.na(T2))
    
    mzData <- subset(wide_df, ZygosityGT1 == "MZ", select = c("T1", "T2"))
    dzData <- subset(wide_df, ZygosityGT1 == "DZ", select = c("T1", "T2"))

    # Define Matrices and Shared Core Algebras
    a <- mxMatrix(type="Full", nrow=1, ncol=1, free=TRUE, values=.6, label="a11", name="a")
    c <- mxMatrix(type="Full", nrow=1, ncol=1, free=TRUE, values=.3, label="c11", name="c")
    e <- mxMatrix(type="Full", nrow=1, ncol=1, free=TRUE, values=.6, label="e11", name="e")
    
    A <- mxAlgebra(expression=a %*% t(a), name="A")
    C <- mxAlgebra(expression=c %*% t(c), name="C")
    E <- mxAlgebra(expression=e %*% t(e), name="E")
    
    # Standardized Proportions for Results & CIs
    propA <- mxAlgebra(expression= A / (A+C+E), name="propA")
    propC <- mxAlgebra(expression= C / (A+C+E), name="propC")
    propE <- mxAlgebra(expression= E / (A+C+E), name="propE")
    
    # Expected Covariance Algebras
    covMZ <- mxAlgebra(expression= rbind( cbind(A+C+E, A+C), cbind(A+C, A+C+E)), name="expCovMZ")
    covDZ <- mxAlgebra(expression= rbind( cbind(A+C+E, 0.5*A+C), cbind(0.5*A+C, A+C+E)), name="expCovDZ")
    meanG <- mxMatrix(type="Full", nrow=1, ncol=2, free=FALSE, values=0, name="expMean")

    # DEFINE SUB-MODELS (Incorporating Namespace Fix)
    modelMZ <- mxModel("MZ", mxData(observed=mzData, type="raw"), 
                       a, c, e, A, C, E, covMZ, meanG,
                       mxExpectationNormal(covariance="expCovMZ", means="expMean", dimnames=c("T1","T2")), 
                       mxFitFunctionML())
    
    modelDZ <- mxModel("DZ", mxData(observed=dzData, type="raw"), 
                       a, c, e, A, C, E, covDZ, meanG,
                       mxExpectationNormal(covariance="expCovDZ", means="expMean", dimnames=c("T1","T2")), 
                       mxFitFunctionML())
    
    # Parent Model Assembly
    ace_model <- mxModel("ACE", 
                         a, c, e, A, C, E, propA, propC, propE,
                         modelMZ, modelDZ, 
                         mxFitFunctionMultigroup(c("MZ","DZ")),
                         mxCI(c("propA", "propC", "propE")))

    # RUN AND EXTRACT 
    tryCatch({
      # Run ACE with Confidence Intervals
      fit_ace <- mxRun(ace_model, intervals=TRUE, silent=TRUE)
      
      # Build nested AE Model (Drop C path) for LRT comparison
      ae_model <- mxModel(fit_ace, name="AE")
      ae_model <- omxSetParameters(ae_model, labels="c11", free=FALSE, values=0)
      fit_ae <- mxRun(ae_model, silent=TRUE)
      
      # Perform Likelihood Ratio Test (LRT)
      lrt <- mxCompare(fit_ace, fit_ae)
      lrt_p <- lrt$p[2]
      
      # Extract Estimates
      est_A <- mxEval(propA, fit_ace)[1,1] * 100
      est_C <- mxEval(propC, fit_ace)[1,1] * 100
      est_E <- mxEval(propE, fit_ace)[1,1] * 100
      
      # Safely extract Confidence Intervals
      cis <- fit_ace$output$confidenceIntervals
      ci_A_L <- ifelse(!is.null(cis), cis["ACE.propA[1,1]", "lbound"] * 100, NA)
      ci_A_U <- ifelse(!is.null(cis), cis["ACE.propA[1,1]", "ubound"] * 100, NA)
      ci_C_L <- ifelse(!is.null(cis), cis["ACE.propC[1,1]", "lbound"] * 100, NA)
      ci_C_U <- ifelse(!is.null(cis), cis["ACE.propC[1,1]", "ubound"] * 100, NA)
      ci_E_L <- ifelse(!is.null(cis), cis["ACE.propE[1,1]", "lbound"] * 100, NA)
      ci_E_U <- ifelse(!is.null(cis), cis["ACE.propE[1,1]", "ubound"] * 100, NA)
      
      results_list[[var]] <- data.frame(
        Metric = var,
        A_pct = est_A, A_lbound = ci_A_L, A_ubound = ci_A_U,
        C_pct = est_C, C_lbound = ci_C_L, C_ubound = ci_C_U,
        E_pct = est_E, E_lbound = ci_E_L, E_ubound = ci_E_U,
        LRT_p_value = lrt_p,
        Status = "Success"
      )
      cat("  [OK] Successfully extracted A, C, E, CIs, and LRT.\n")
    }, error = function(err) {
      cat("  [!] Error in variable:", var, "-", conditionMessage(err), "\n")
    })
  }
  
  # --- STEP C: COMPILE, FDR CORRECT, AND SAVE ---
  final_table <- do.call(rbind, results_list)
  if(!is.null(final_table)) {
    # Apply FDR correction to the LRT p-values
    final_table$FDR_q_value <- p.adjust(final_table$LRT_p_value, method="fdr")
    
    # Organize final layout
    final_table <- final_table %>% 
      select(Metric, starts_with("A"), starts_with("C"), starts_with("E"), 
             LRT_p_value, FDR_q_value, Status)
             
    out_file <- paste0(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "ace_results_"), gender_label, "_FDR.csv")
    write.csv(final_table, out_file, row.names=FALSE)
    cat(paste0("\nSaved FDR-corrected results to: ", out_file, "\n"))
  }
}

# EXECUTION
# Gender 0 = n=134 | Gender 1 = n=104
run_stratified_ace(df_raw %>% filter(Gender == 0), "gender0")
run_stratified_ace(df_raw %>% filter(Gender == 1), "gender1")

cat("\nALL SEX-STRATIFIED ACE MODELS COMPLETED.\n")