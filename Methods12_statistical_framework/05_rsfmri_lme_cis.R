# rsfMRI_LME_CIs.R
# Extracts proper 95% CIs for rsfMRI population LME results
# Uses the same model specification as the original analysis
#
# PROVENANCE NOTE (important -- read before using): this script originally
# computed CIs across BOTH static and dynamic-fluidity rsfMRI metrics
# together (the full 12-metric list below). It was LATER CUT to static
# metrics only -- dynamic fluidity was moved to its own dedicated pipeline
# using the newly-adopted window=100/step=20 methodology (see
# DynamicFluidity.py and the dynfc_robustness_battery/ scripts), which
# this script's dynamic-metric rows predate and do NOT reflect. If reusing
# this script, restrict the `metrics` vector to the static entries only
# (Olfactory_FC_Cortical, DMN_Segregation_Cortical,
# Salience_Segregation_Cortical, Salience_FC_Cortical, Reward_FC_Cortical,
# DMN_FC_Cortical, Inter_FC_OlfPrimary_RewGateway) -- the Dynamic_Fluidity_*
# rows are kept below only for historical completeness and should not be
# trusted as reflecting the current, reported dynamic-fluidity methodology.

library(nlme)
library(dplyr)

master <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv"),
                   stringsAsFactors=FALSE)
rsf    <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "rsfMRI_Tier2_Decomposed_Metrics.csv"),
                   stringsAsFactors=FALSE)
dfc    <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "rsfMRI_Tier3_Dynamic_Metrics.csv"),
                   stringsAsFactors=FALSE)

df <- master %>%
    select(Subject, TwinPairID, ZygosityGT1, Severity,
           Age_in_Yrs, Gender, SSAGA_TB_Still_Smoking) %>%
    left_join(rsf, by='Subject') %>%
    left_join(dfc, by='Subject')

df$Gender_Label <- factor(df$Gender, levels=c(0,1),
                           labels=c('Female','Male'))
df$Gender_Label <- relevel(df$Gender_Label, ref='Female')

# NOTE: static metrics only are current/trusted -- see provenance note above.
# Dynamic_Fluidity_* rows retained for historical record only.
metrics <- c(
    'Olfactory_FC_Cortical',
    'DMN_Segregation_Cortical',
    'Salience_Segregation_Cortical',
    'Dynamic_Fluidity_Reward',
    'Dynamic_Fluidity_DMN',
    'Dynamic_Fluidity_Salience',
    'Global_Dynamic_Fluidity',
    'Salience_FC_Cortical',
    'Reward_FC_Cortical',
    'DMN_FC_Cortical',
    'Inter_FC_OlfPrimary_RewGateway',
    'Dynamic_Fluidity_Olfactory'
)

results <- data.frame()
for (metric in metrics) {
    if (!metric %in% colnames(df)) {
        cat(sprintf("  MISSING: %s\n", metric))
        next
    }
    df$Y    <- df[[metric]]
    df_fit  <- df[!is.na(df$Y), ]

    tryCatch({
        mod <- lme(
            Y ~ Severity + Age_in_Yrs + Gender_Label +
                SSAGA_TB_Still_Smoking,
            random  = ~1 | TwinPairID,
            data    = df_fit,
            method  = 'ML',
            control = lmeControl(opt='optim', maxIter=200)
        )
        ci  <- intervals(mod, which='fixed')$fixed
        sev <- summary(mod)$tTable['Severity',]

        sd_y   <- sd(df_fit$Y,        na.rm=TRUE)
        sd_sev <- sd(df_fit$Severity, na.rm=TRUE)

        results <- rbind(results, data.frame(
            Metric            = metric,
            Standardized_Beta = round(sev['Value']  * sd_sev / sd_y, 4),
            CI_lower          = round(ci['Severity','lower'] * sd_sev / sd_y, 4),
            CI_upper          = round(ci['Severity','upper'] * sd_sev / sd_y, 4),
            P_Value           = round(sev['p-value'], 4),
            N                 = nrow(df_fit)
        ))
        cat(sprintf("  %s: beta=%.3f [%.3f, %.3f] p=%.3f\n",
                    metric,
                    sev['Value']*sd_sev/sd_y,
                    ci['Severity','lower']*sd_sev/sd_y,
                    ci['Severity','upper']*sd_sev/sd_y,
                    sev['p-value']))
    }, error=function(e) {
        cat(sprintf("  ERROR %s: %s\n", metric, e$message))
    })
}

results$FDR_q <- p.adjust(results$P_Value, method='BH')
print(results)
write.csv(results,
          file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "rsfMRI_LME_CIs.csv"),
          row.names=FALSE)
cat("\nSaved: rsfMRI_LME_CIs.csv\n")
