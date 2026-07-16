# Mundlak_DynamicFC_MotionControlled.R
# Within/between decomposition of dynamic FC fluidity
# with Mean_FD added as motion covariate for reviewer robustness

library(nlme)
library(dplyr)

master <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv"),
                   stringsAsFactors=FALSE)
dfc    <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "final_stats", "rsfMRI_Tier3_Dynamic_Metrics.csv"),
                   stringsAsFactors=FALSE)
motion <- read.csv(file.path(Sys.getenv("PROJECT_ROOT", "."), "rsfMRI_motion_FD.csv"),
                   stringsAsFactors=FALSE)

df <- master %>%
    select(Subject, TwinPairID, ZygosityGT1, Severity,
           Age_in_Yrs, Gender, SSAGA_TB_Still_Smoking) %>%
    left_join(dfc,    by='Subject') %>%
    left_join(motion, by='Subject')

cat(sprintf("N after merge: %d\n", nrow(df)))
cat(sprintf("Motion data available: %d subjects\n", sum(!is.na(df$Mean_FD))))

df$Gender_Label <- factor(df$Gender, levels=c(0,1),
                           labels=c('Female','Male'))
df$Gender_Label <- relevel(df$Gender_Label, ref='Female')

pair_means     <- df %>%
    group_by(TwinPairID) %>%
    summarise(Severity_BF = mean(Severity, na.rm=TRUE), .groups='drop')
df             <- merge(df, pair_means, by='TwinPairID')
df$Severity_WF <- df$Severity - df$Severity_BF

metrics <- c('Global_Dynamic_Fluidity',
             'Dynamic_Fluidity_Reward',
             'Dynamic_Fluidity_Salience',
             'Dynamic_Fluidity_DMN',
             'Dynamic_Fluidity_Olfactory')

results <- data.frame()

for (metric in metrics) {
    if (!metric %in% colnames(df)) {
        cat(sprintf("  MISSING: %s\n", metric))
        next
    }
    df$Y     <- df[[metric]]
    df_clean <- df[!is.na(df$Y) & !is.na(df$Mean_FD), ]
    cat(sprintf("\n  %s (N=%d)...\n", metric, nrow(df_clean)))

    tryCatch({
        mod <- lme(
            Y ~ Severity_BF + Severity_WF +
                Age_in_Yrs + Gender_Label +
                SSAGA_TB_Still_Smoking + Mean_FD,
            random  = ~1 | TwinPairID,
            data    = df_clean,
            method  = 'ML',
            control = lmeControl(opt='optim',
                                 maxIter=200,
                                 msMaxIter=200)
        )

        coefs  <- summary(mod)$tTable
        bf_row <- coefs['Severity_BF',]
        wf_row <- coefs['Severity_WF',]
        fd_row <- coefs['Mean_FD',]

        sd_y  <- sd(df_clean$Y,           na.rm=TRUE)
        sd_bf <- sd(df_clean$Severity_BF, na.rm=TRUE)
        sd_wf <- sd(df_clean$Severity_WF, na.rm=TRUE)
        sd_fd <- sd(df_clean$Mean_FD,     na.rm=TRUE)

        results <- rbind(results, data.frame(
            Metric   = metric,
            Beta_BF  = round(bf_row['Value'] * sd_bf / sd_y, 4),
            P_BF     = round(bf_row['p-value'], 4),
            Beta_WF  = round(wf_row['Value'] * sd_wf / sd_y, 4),
            P_WF     = round(wf_row['p-value'], 4),
            Beta_FD  = round(fd_row['Value'] * sd_fd / sd_y, 4),
            P_FD     = round(fd_row['p-value'], 4),
            N        = nrow(df_clean)
        ))

        cat(sprintf("    BF: beta=%.3f (p=%.3f)\n",
                    bf_row['Value']*sd_bf/sd_y, bf_row['p-value']))
        cat(sprintf("    WF: beta=%.3f (p=%.3f)\n",
                    wf_row['Value']*sd_wf/sd_y, wf_row['p-value']))
        cat(sprintf("    FD: beta=%.3f (p=%.3f) [motion effect]\n",
                    fd_row['Value']*sd_fd/sd_y, fd_row['p-value']))

    }, error=function(e) {
        cat(sprintf("    ERROR: %s\n", e$message))
    })
}

cat("\n=== FULL RESULTS ===\n")
results$FDR_BF <- p.adjust(results$P_BF, method='BH')
results$FDR_WF <- p.adjust(results$P_WF, method='BH')
print(results, digits=4)

write.csv(results,
          file.path(Sys.getenv("PROJECT_ROOT", "."), "mundlak_DynamicFC_MotionControlled.csv"),
          row.names=FALSE)
cat("\nSaved: mundlak_DynamicFC_MotionControlled.csv\n")
