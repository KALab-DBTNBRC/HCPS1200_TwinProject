# NCT Q2: Per-node ACE heritability of controllability
# B1-matched ACE on all four points (profile CIs, omxSetParameters AE,
# scaled residuals, fixed zero means).
#
# Three model comparisons per node:
#   LRT_C_p  : ACE vs AE  (df=1) -- is shared environment needed?
#   LRT_A_p  : AE  vs E   (df=1) -- is heritability significant?
#   FDR_q_C / FDR_q_A : BH correction within metric (360 tests each)

suppressPackageStartupMessages({
  library(OpenMx)
  library(parallel)
  library(dplyr)
})

CTRL_DIR  <- file.path(Sys.getenv("PROJECT_ROOT", "."), "NCT_inputs/Controllability")
R_INPUTS  <- file.path(CTRL_DIR, "r_inputs")
NCT_DIR   <- file.path(Sys.getenv("PROJECT_ROOT", "."), "NCT_inputs")
BEH_CSV   <- file.path(Sys.getenv("PROJECT_ROOT", "."), "twintables/network_roi_metrics_FINAL.csv")
OUT_DIR   <- file.path(CTRL_DIR, "heritability")
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

MC_CORES   <- 40L
MX_THREADS <- 1L
ALPHA      <- 0.05

cat(sprintf("[%s] NCT Q2: Controllability Heritability\n", format(Sys.time(), "%H:%M:%S")))

mxOption(NULL, "Number of Threads", MX_THREADS)
mxOption(NULL, "Default optimizer",  "SLSQP")

if (!file.exists(file.path(R_INPUTS, "MIND_AC_T3.csv"))) {
  cat(sprintf("[%s] Exporting .npy -> CSV...\n", format(Sys.time(), "%H:%M:%S")))
  if (system("python3 nct_export_for_r.py") != 0)
    stop("nct_export_for_r.py failed.")
}

cat(sprintf("[%s] Loading arrays...\n", format(Sys.time(), "%H:%M:%S")))

load_csv <- function(name) {
  arr <- as.matrix(read.table(file.path(R_INPUTS, paste0(name, ".csv")),
                               sep=",", header=FALSE, colClasses="numeric"))
  arr
}

metrics <- list(
  MIND_AC_T3 = load_csv("MIND_AC_T3"),
  SC_AC_T3   = load_csv("SC_AC_T3"),
  MIND_MC    = load_csv("MIND_MC"),
  SC_MC      = load_csv("SC_MC")
)

subj_idx   <- read.csv(file.path(CTRL_DIR, "subject_index.csv"))
node_df    <- read.csv(file.path(NCT_DIR,  "node_names.csv"))
beh        <- read.csv(BEH_CSV, stringsAsFactors=FALSE)
node_names <- node_df$Region
N_NODES    <- length(node_names)

subj_idx$subj_id <- as.integer(subj_idx$subj_id)
beh$Subject      <- as.integer(beh$Subject)

meta <- subj_idx %>%
  left_join(beh %>% select(Subject, TwinPairID, ZygosityGT1,
                             Age_in_Yrs, Gender, Severity),
            by=c("subj_id"="Subject"))

stopifnot(nrow(metrics[[1]]) == nrow(meta))

pair_df <- meta %>%
  filter(!is.na(TwinPairID)) %>%
  group_by(TwinPairID) %>%
  arrange(subj_id) %>%
  filter(n() == 2) %>%
  summarise(
    subj_T1  = subj_id[1], subj_T2 = subj_id[2],
    row_T1   = row[1],     row_T2  = row[2],
    ZygLabel = ZygosityGT1[1],
    .groups  = "drop"
  ) %>%
  mutate(ZygCode = ifelse(grepl("^MZ", ZygLabel, ignore.case=TRUE), "MZ", "DZ"))

n_mz <- sum(pair_df$ZygCode == "MZ")
n_dz <- sum(pair_df$ZygCode == "DZ")
cat(sprintf("  Pairs: %d  (MZ=%d  DZ=%d)\n", nrow(pair_df), n_mz, n_dz))
if (n_mz < 20 || n_dz < 10) stop("Insufficient pairs. Check ZygosityGT1 coding.")

residualise_and_scale <- function(arr, meta_df) {
  apply(arr, 2, function(y) {
    df  <- data.frame(y=y, age=meta_df$Age_in_Yrs, sex=meta_df$Gender)
    ok  <- complete.cases(df)
    out <- rep(NA_real_, length(y))
    if (sum(ok) < 10) return(out)
    fit     <- lm(y ~ age + sex, data=df[ok, ])
    out[ok] <- as.numeric(scale(residuals(fit)))
    out
  })
}

fit_ace_node <- function(node_vals, pair_df_local) {

  T1 <- node_vals[pair_df_local$row_T1 + 1L]
  T2 <- node_vals[pair_df_local$row_T2 + 1L]
  zy <- pair_df_local$ZygCode

  keep <- !is.na(T1) & !is.na(T2)
  T1 <- T1[keep]; T2 <- T2[keep]; zy <- zy[keep]

  mz  <- zy == "MZ";  dz <- zy == "DZ"
  rMZ <- if (sum(mz) > 2) cor(T1[mz], T2[mz]) else NA_real_
  rDZ <- if (sum(dz) > 2) cor(T1[dz], T2[dz]) else NA_real_

  mzData <- data.frame(T1=T1[mz], T2=T2[mz])
  dzData <- data.frame(T1=T1[dz], T2=T2[dz])

  null_result <- function(s) list(
    rMZ=rMZ, rDZ=rDZ,
    A_pct=NA_real_, A_lbound=NA_real_, A_ubound=NA_real_,
    C_pct=NA_real_, C_lbound=NA_real_, C_ubound=NA_real_,
    E_pct=NA_real_, E_lbound=NA_real_, E_ubound=NA_real_,
    LRT_C_p=NA_real_, LRT_A_p=NA_real_, Status=s)

  a     <- mxMatrix("Full", 1, 1, TRUE,  .6, "a11", name="a")
  c     <- mxMatrix("Full", 1, 1, TRUE,  .3, "c11", name="c")
  e     <- mxMatrix("Full", 1, 1, TRUE,  .6, "e11", name="e")
  A     <- mxAlgebra(a %*% t(a),         name="A")
  C     <- mxAlgebra(c %*% t(c),         name="C")
  E     <- mxAlgebra(e %*% t(e),         name="E")
  propA <- mxAlgebra(A / (A+C+E),        name="propA")
  propC <- mxAlgebra(C / (A+C+E),        name="propC")
  propE <- mxAlgebra(E / (A+C+E),        name="propE")
  covMZ <- mxAlgebra(rbind(cbind(A+C+E, A+C),     cbind(A+C,     A+C+E)), name="expCovMZ")
  covDZ <- mxAlgebra(rbind(cbind(A+C+E, 0.5*A+C), cbind(0.5*A+C, A+C+E)), name="expCovDZ")
  meanG <- mxMatrix("Full", 1, 2, FALSE, 0, name="expMean")

  modelMZ <- mxModel("MZ", mxData(mzData, "raw"),
    a, c, e, A, C, E, covMZ, meanG,
    mxExpectationNormal("expCovMZ", "expMean", dimnames=c("T1","T2")),
    mxFitFunctionML())

  modelDZ <- mxModel("DZ", mxData(dzData, "raw"),
    a, c, e, A, C, E, covDZ, meanG,
    mxExpectationNormal("expCovDZ", "expMean", dimnames=c("T1","T2")),
    mxFitFunctionML())

  ace_model <- mxModel("ACE",
    a, c, e, A, C, E, propA, propC, propE,
    modelMZ, modelDZ,
    mxFitFunctionMultigroup(c("MZ","DZ")),
    mxCI(c("propA","propC","propE")))

  fit_ace <- tryCatch(
    mxRun(ace_model, intervals=TRUE, silent=TRUE), error=function(err) NULL)
  if (is.null(fit_ace)) return(null_result("FAILED_ACE"))

  ace_code <- fit_ace$output$status$code
  if (!(ace_code %in% c(0L, 1L)))
    return(null_result(sprintf("CONVERGE_FAIL_code%d", ace_code)))

  est_A <- mxEval(propA, fit_ace)[1,1]
  est_C <- mxEval(propC, fit_ace)[1,1]
  est_E <- mxEval(propE, fit_ace)[1,1]

  cis    <- fit_ace$output$confidenceIntervals
  get_ci <- function(nm, bd) {
    key <- paste0("ACE.", nm, "[1,1]")
    if (!is.null(cis) && key %in% rownames(cis)) cis[key, bd] else NA_real_
  }

  fit_ae <- tryCatch({
    ae <- mxModel(fit_ace, name="AE")
    ae <- omxSetParameters(ae, labels="c11", free=FALSE, values=0)
    mxRun(ae, silent=TRUE)
  }, error=function(err) NULL)

  lrt_C_p <- NA_real_
  if (!is.null(fit_ae) && fit_ae$output$status$code %in% c(0L, 1L)) {
    lrt_C   <- mxCompare(fit_ace, fit_ae)
    lrt_C_p <- lrt_C$p[2]
  }

  lrt_A_p <- NA_real_
  if (!is.null(fit_ae) && fit_ae$output$status$code %in% c(0L, 1L)) {
    fit_e <- tryCatch({
      e_mod <- mxModel(fit_ae, name="E")
      e_mod <- omxSetParameters(e_mod, labels="a11", free=FALSE, values=0)
      mxRun(e_mod, silent=TRUE)
    }, error=function(err) NULL)

    if (!is.null(fit_e) && fit_e$output$status$code %in% c(0L, 1L)) {
      lrt_A   <- mxCompare(fit_ae, fit_e)
      lrt_A_p <- lrt_A$p[2]
    }
  }

  list(
    rMZ      = round(rMZ,    4),
    rDZ      = round(rDZ,    4),
    A_pct    = round(est_A,  4),
    A_lbound = round(get_ci("propA","lbound"), 4),
    A_ubound = round(get_ci("propA","ubound"), 4),
    C_pct    = round(est_C,  4),
    C_lbound = round(get_ci("propC","lbound"), 4),
    C_ubound = round(get_ci("propC","ubound"), 4),
    E_pct    = round(est_E,  4),
    E_lbound = round(get_ci("propE","lbound"), 4),
    E_ubound = round(get_ci("propE","ubound"), 4),
    LRT_C_p  = round(lrt_C_p, 6),
    LRT_A_p  = round(lrt_A_p, 6),
    Status   = if (ace_code == 0L) "OK" else "CONVERGE_WARN"
  )
}

all_results <- list()

for (metric_name in names(metrics)) {
  cat(sprintf("\n[%s] -- %s --\n", format(Sys.time(), "%H:%M:%S"), metric_name))

  arr_resid <- residualise_and_scale(metrics[[metric_name]], meta)

  t0 <- proc.time()
  pair_snap <- pair_df

  node_results <- parallel::mclapply(
    seq_len(N_NODES),
    function(node_i) {
      mxOption(NULL, "Number of Threads", MX_THREADS)
      tryCatch(
        fit_ace_node(arr_resid[, node_i], pair_snap),
        error=function(err) list(
          rMZ=NA_real_, rDZ=NA_real_,
          A_pct=NA_real_, A_lbound=NA_real_, A_ubound=NA_real_,
          C_pct=NA_real_, C_lbound=NA_real_, C_ubound=NA_real_,
          E_pct=NA_real_, E_lbound=NA_real_, E_ubound=NA_real_,
          LRT_C_p=NA_real_, LRT_A_p=NA_real_,
          Status=paste0("R_ERROR: ", conditionMessage(err))))
    },
    mc.cores=MC_CORES, mc.set.seed=FALSE
  )
  elapsed <- (proc.time()-t0)["elapsed"]

  res_df <- as.data.frame(do.call(rbind, lapply(node_results, function(x) {
    x[sapply(x, is.null)] <- NA
    as.data.frame(lapply(x, function(v) if (length(v)==0) NA else v[1]))
  })))

  res_df$Region <- node_names
  res_df$Metric <- metric_name

  res_df$FDR_q_C   <- p.adjust(as.numeric(res_df$LRT_C_p), method="fdr")
  res_df$FDR_q_A   <- p.adjust(as.numeric(res_df$LRT_A_p), method="fdr")
  res_df$Sig_C_FDR <- res_df$FDR_q_C < ALPHA
  res_df$Sig_A_FDR <- res_df$FDR_q_A < ALPHA
  res_df$Anomalous_DZ <- as.numeric(res_df$rDZ) > as.numeric(res_df$rMZ)

  res_df <- res_df[, c("Region","Metric","rMZ","rDZ",
                        "A_pct","A_lbound","A_ubound",
                        "C_pct","C_lbound","C_ubound",
                        "E_pct","E_lbound","E_ubound",
                        "LRT_C_p","FDR_q_C","Sig_C_FDR",
                        "LRT_A_p","FDR_q_A","Sig_A_FDR",
                        "Status", "Anomalous_DZ")]

  out_path <- file.path(OUT_DIR, sprintf("nct_heritability_%s.csv", metric_name))
  write.csv(res_df, out_path, row.names=FALSE)
  all_results[[metric_name]] <- res_df
}

combined <- do.call(rbind, all_results)
write.csv(combined, file.path(OUT_DIR, "nct_heritability_combined.csv"), row.names=FALSE)

cat(sprintf("[%s] Q2 complete -> %s/\n", format(Sys.time(),"%H:%M:%S"), OUT_DIR))
