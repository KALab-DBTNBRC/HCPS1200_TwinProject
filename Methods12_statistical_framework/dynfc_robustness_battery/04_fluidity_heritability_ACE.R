# dynfc_fluidity_ACE.R
# -------------------------------------------------------------------
# Univariate ACE heritability of dynamic FC fluidity (and, if present,
# surrogate-corrected excess fluidity). Substantiates the genetic reading:
# a genetic severity-fluidity link requires fluidity itself to be heritable.
#
# Output (kept as record), matching the lab Phase5A convention:
#   dynfc_fluidity_ACE.csv  with A_pct / C_pct / E_pct (+95% CIs),
#                           LRT_p_value (drop-C), FDR_q_value, Status
#
# Run:  Rscript dynfc_fluidity_ACE.R
# Deps: OpenMx, dplyr, tidyr
# -------------------------------------------------------------------
suppressMessages({library(OpenMx); library(dplyr); library(tidyr)})
mxOption(NULL, "Default optimizer", "SLSQP")

BASE   <- Sys.getenv("PROJECT_ROOT", ".")
MASTER <- file.path(BASE, "MASTER_ROI_METRICS_DTI_FBA.csv")
# Set METRIC_FILE=rsfMRI_Tier3_Dynamic_Excess.csv to run on excess fluidity.
DFC    <- Sys.getenv("METRIC_FILE", file.path(BASE, "rsfMRI_Tier3_Dynamic_Metrics.csv"))
OUT    <- file.path(BASE, sprintf("dynfc_fluidity_ACE%s.csv",
                                  ifelse(grepl("Excess", DFC), "_excess", "")))

METRICS <- c("Global_Dynamic_Fluidity","Dynamic_Fluidity_Reward",
             "Dynamic_Fluidity_Salience","Dynamic_Fluidity_DMN",
             "Dynamic_Fluidity_Olfactory")

m   <- read.csv(MASTER, stringsAsFactors = FALSE)
d   <- read.csv(DFC,    stringsAsFactors = FALSE)
dat <- merge(m[, c("Subject","TwinPairID","ZygosityGT1","Age_in_Yrs",
                   "Gender","SSAGA_TB_Still_Smoking")], d, by = "Subject")

ace_one <- function(metric) {
  df <- dat[!is.na(dat[[metric]]), ]
  # residualize on covariates, then standardize
  df$resid <- scale(resid(lm(df[[metric]] ~ Age_in_Yrs + factor(Gender) +
                               SSAGA_TB_Still_Smoking, data = df)))[, 1]
  # order twins within pair, reshape wide
  df <- df %>% group_by(TwinPairID) %>% mutate(ord = row_number()) %>%
        ungroup() %>% filter(ord <= 2)
  zyg <- df %>% group_by(TwinPairID) %>% summarise(z = first(ZygosityGT1), .groups="drop")
  wide <- df %>% select(TwinPairID, ord, resid) %>%
          pivot_wider(names_from = ord, values_from = resid,
                      names_prefix = "t") %>% inner_join(zyg, by = "TwinPairID")
  wide <- wide[!is.na(wide$t1) & !is.na(wide$t2), ]
  mz <- as.data.frame(wide[wide$z == "MZ", c("t1","t2")])
  dz <- as.data.frame(wide[wide$z == "DZ", c("t1","t2")])
  sel <- c("t1","t2")

  ace <- mxModel("ACE",
    mxMatrix("Lower", 1, 1, TRUE, .6, "a11", name="a"),
    mxMatrix("Lower", 1, 1, TRUE, .6, "c11", name="c"),
    mxMatrix("Lower", 1, 1, TRUE, .6, "e11", name="e"),
    mxAlgebra(a %*% t(a), name="A"),
    mxAlgebra(c %*% t(c), name="C"),
    mxAlgebra(e %*% t(e), name="E"),
    mxAlgebra(A + C + E, name="V"),
    mxMatrix("Full", 1, 1, FALSE, 0, name="meanG"),
    mxModel("MZ",
      mxData(mz, "raw"),
      mxAlgebra(rbind(cbind(ACE.V, ACE.A + ACE.C),
                      cbind(ACE.A + ACE.C, ACE.V)), name="expCov"),
      mxAlgebra(cbind(ACE.meanG, ACE.meanG), name="expMean"),
      mxExpectationNormal("expCov", "expMean", sel),
      mxFitFunctionML()),
    mxModel("DZ",
      mxData(dz, "raw"),
      mxAlgebra(rbind(cbind(ACE.V, 0.5 %x% ACE.A + ACE.C),
                      cbind(0.5 %x% ACE.A + ACE.C, ACE.V)), name="expCov"),
      mxAlgebra(cbind(ACE.meanG, ACE.meanG), name="expMean"),
      mxExpectationNormal("expCov", "expMean", sel),
      mxFitFunctionML()),
    mxFitFunctionMultigroup(c("MZ","DZ")),
    mxCI(c("h2","c2","e2")),
    mxAlgebra(A/V, name="h2"), mxAlgebra(C/V, name="c2"), mxAlgebra(E/V, name="e2"))

  fit  <- tryCatch(mxRun(mxModel(ace, mxCI(c("h2","c2","e2"))),
                         intervals=TRUE, silent=TRUE), error=function(e) NULL)
  if (is.null(fit)) return(NULL)
  s   <- summary(fit)
  ci  <- s$CI
  ae  <- mxRun(omxSetParameters(ace, labels="c11", free=FALSE, values=0),
               silent=TRUE)                              # drop-C
  lrt <- mxCompare(fit, ae)$p[2]
  data.frame(
    Metric      = metric,
    A_pct       = round(mxEval(h2, fit)[1,1], 4),
    A_lbound    = round(ci["ACE.h2[1,1]","lbound"], 4),
    A_ubound    = round(ci["ACE.h2[1,1]","ubound"], 4),
    C_pct       = round(mxEval(c2, fit)[1,1], 4),
    C_lbound    = round(ci["ACE.c2[1,1]","lbound"], 4),
    C_ubound    = round(ci["ACE.c2[1,1]","ubound"], 4),
    E_pct       = round(mxEval(e2, fit)[1,1], 4),
    E_lbound    = round(ci["ACE.e2[1,1]","lbound"], 4),
    E_ubound    = round(ci["ACE.e2[1,1]","ubound"], 4),
    LRT_p_value = round(lrt, 4),
    Status      = s$statusCode
  )
}

res <- do.call(rbind, lapply(METRICS, function(mt) {
  cat("  ACE:", mt, "\n"); ace_one(mt)
}))
res$FDR_q_value <- round(p.adjust(res$LRT_p_value, method="BH"), 4)
res <- res[, c("Metric","A_pct","A_lbound","A_ubound","C_pct","C_lbound","C_ubound",
               "E_pct","E_lbound","E_ubound","LRT_p_value","FDR_q_value","Status")]
write.csv(res, OUT, row.names=FALSE)
cat("\nSaved:", OUT, "\n"); print(res)
