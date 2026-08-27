# ==============================================================================
# STUDY 2: FROZEN LOGISTIC vs FROZEN XGBOOST MODEL COMPARISON
# Repository folder: loan-default-risk-advanced/
# File: model_comparison.R
#
# XGBoost source file used to create the frozen challenger:
#   loan_default_xgb.R
#
# PURPOSE
# -------
# Compare the already-frozen FULL Logistic Regression baseline against the
# already-frozen XGBoost challenger on the EXACT SAME 9,999 holdout loans.
#
# THIS SCRIPT DOES NOT:
#   - modify the Logistic Regression study;
#   - retrain Logistic Regression;
#   - retrain or retune XGBoost;
#   - change XGBoost calibration;
#   - change the XGBoost research threshold;
#   - build an ensemble.
#
# PRIMARY COMPARISON
# ------------------
# 1) Probability discrimination:
#      ROC-AUC, PR-AUC
# 2) Probability quality:
#      Brier score, log loss
# 3) Paired statistical testing:
#      DeLong test for ROC-AUC
#      Paired stratified bootstrap for PR-AUC difference
# 4) Thresholded classification at the SAME conventional threshold 0.50:
#      Recall, specificity, precision, F1, balanced accuracy, McNemar test
#
# XGBoost's development-selected research threshold is also reported, but it is
# DESCRIPTIVE ONLY because the frozen Logistic baseline does not have a
# comparable OOF/development-selected threshold. It must NOT be used to claim
# algorithmic superiority over Logistic Regression.
#
# PRIMARY LOGISTIC BASELINE
# -------------------------
# The thesis/reproducibility FULL Logistic model is used here, because the
# published holdout metrics (AUC ~0.751, TN=8781, FN=1133, FP=39, TP=46) came
# from the full model. The stepwise model used by the deployed Study 1 Shiny app
# is intentionally NOT substituted into this primary scientific comparison.
# ==============================================================================


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

DATA_FILE <- "Loan_default.csv"

EXPECTED_MD5 <- "5f5a3364753b86ef3472cba30af224ce"

# Frozen Study 1 model artifact produced by the reproducibility script.
LOGISTIC_MODEL_FILE <- file.path(
  "thesis_reproducibility_outputs",
  "models",
  "full_logistic_model.rds"
)

# Updated XGBoost source filename (audit only; this script does NOT source it).
XGB_SOURCE_FILE <- "loan_default_xgb.R"

# Frozen XGBoost keyed final-test predictions created by loan_default_xgb.R.
XGB_PREDICTIONS_FILE <- file.path(
  "xgboost_study2_outputs",
  "tables",
  "xgb_locked_test_keyed_predictions.csv"
)

# Frozen XGBoost threshold selected from development OOF predictions.
XGB_THRESHOLD_FILE <- file.path(
  "xgboost_study2_outputs",
  "tables",
  "xgb_selected_research_threshold.csv"
)

OUTPUT_DIR <- "model_comparison_outputs"
TABLE_DIR  <- file.path(OUTPUT_DIR, "tables")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
META_DIR   <- file.path(OUTPUT_DIR, "metadata")

# Paired stratified bootstrap.
N_BOOT <- 2000L
BOOT_SEED <- 20260827L

# Exact Study 1 reproducibility seeds.
SEED_SAMPLE <- 1L
SEED_SPLIT  <- 123L

ALPHA <- 0.05


# ==============================================================================
# 1. PACKAGES AND OUTPUT FOLDERS
# ==============================================================================

required_packages <- c(
  "tidyverse",
  "caret",
  "pROC",
  "PRROC"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required package(s): ",
      paste(missing_packages, collapse = ", "),
      "\nInstall first with:\ninstall.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))"
    )
  )
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(pROC)
  library(PRROC)
})

dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(TABLE_DIR, showWarnings = FALSE)
dir.create(FIGURE_DIR, showWarnings = FALSE)
dir.create(META_DIR, showWarnings = FALSE)

capture.output(
  sessionInfo(),
  file = file.path(META_DIR, "sessionInfo.txt")
)

package_versions <- tibble(
  Package = required_packages,
  Version = vapply(
    required_packages,
    function(x) as.character(packageVersion(x)),
    character(1)
  )
)

write_csv(
  package_versions,
  file.path(META_DIR, "package_versions.csv")
)


# ==============================================================================
# 2. REQUIRED FILE CHECKS
# ==============================================================================

required_files <- c(
  DATA_FILE,
  LOGISTIC_MODEL_FILE,
  XGB_PREDICTIONS_FILE,
  XGB_THRESHOLD_FILE
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    paste0(
      "Required frozen artifact(s) not found:\n",
      paste0(" - ", missing_files, collapse = "\n"),
      "\n\nDo NOT retrain a model to bypass this error. ",
      "Locate the existing frozen artifact first."
    )
  )
}

if (!file.exists(XGB_SOURCE_FILE)) {
  warning(
    paste0(
      XGB_SOURCE_FILE,
      " was not found in the current folder. ",
      "This does not prevent comparison because the frozen XGBoost predictions ",
      "are used, but the source-file audit entry will show 'not found'."
    )
  )
}


# ==============================================================================
# 3. VERIFY EXACT DATASET SNAPSHOT
# ==============================================================================

observed_md5 <- unname(tools::md5sum(DATA_FILE))

if (!identical(observed_md5, EXPECTED_MD5)) {
  stop(
    paste0(
      "Dataset fingerprint mismatch.\n",
      "Expected MD5: ", EXPECTED_MD5, "\n",
      "Observed MD5: ", observed_md5, "\n",
      "The comparison must use the exact Study 1 / Study 2 source snapshot."
    )
  )
}

cat("[OK] Dataset MD5 matches the frozen studies.\n")


# ==============================================================================
# 4. RECONSTRUCT THE EXACT 50,000 SAMPLE AND 9,999 HOLDOUT
#
# This is NOT Logistic retraining.
# It only reconstructs the exact paired evaluation population so the already
# frozen Logistic model can generate probabilities for the same loans used by
# the already frozen XGBoost challenger.
# ==============================================================================

raw_data <- read_csv(
  DATA_FILE,
  show_col_types = FALSE
)

stopifnot(
  nrow(raw_data) == 255347,
  ncol(raw_data) == 18,
  sum(is.na(raw_data)) == 0,
  sum(duplicated(raw_data)) == 0
)

set.seed(SEED_SAMPLE)

sample_df <- dplyr::sample_n(
  raw_data,
  size = 50000
)

rm(raw_data)
gc(verbose = FALSE)

# Match Study 1 preprocessing before createDataPartition().
data_with_id <- sample_df %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(Default = factor(Default, levels = c(0, 1)))

# Study 1's modelling data removed LoanID before fitting.
data_clean <- data_with_id %>%
  select(-LoanID)

set.seed(SEED_SPLIT)

train_idx <- createDataPartition(
  data_clean$Default,
  p = 0.80,
  list = FALSE
)

dev_data  <- data_clean[train_idx, ]
test_data <- data_clean[-train_idx, ]

# Preserve the paired LoanID audit key from the same rows.
dev_loan_ids <- as.character(
  data_with_id$LoanID[train_idx]
)

test_loan_ids <- as.character(
  data_with_id$LoanID[-train_idx]
)

stopifnot(
  nrow(dev_data) == 40001,
  nrow(test_data) == 9999,
  length(test_loan_ids) == 9999,
  length(unique(test_loan_ids)) == 9999
)

y_test <- as.numeric(
  as.character(test_data$Default)
)

stopifnot(all(y_test %in% c(0, 1)))

cat("[OK] Exact 40,001 / 9,999 Study 1 partition reconstructed.\n")


# ==============================================================================
# 5. LOAD THE FROZEN FULL LOGISTIC MODEL — NO RETRAINING
# ==============================================================================

full_logistic_model <- readRDS(
  LOGISTIC_MODEL_FILE
)

if (!inherits(full_logistic_model, "glm")) {
  stop(
    "Frozen Logistic artifact is not a glm object."
  )
}

logistic_prob <- predict(
  full_logistic_model,
  newdata = test_data,
  type = "response"
)

logistic_prob <- as.numeric(logistic_prob)

stopifnot(
  length(logistic_prob) == 9999,
  all(is.finite(logistic_prob)),
  all(logistic_prob >= 0 & logistic_prob <= 1)
)

cat("[OK] Frozen FULL Logistic model loaded and scored. No retraining occurred.\n")


# ==============================================================================
# 6. VALIDATE THAT THE FROZEN LOGISTIC RESULT MATCHES THE PROVEN THESIS BASELINE
# ==============================================================================

logistic_class_050 <- ifelse(
  logistic_prob >= 0.50,
  1L,
  0L
)

log_TN <- sum(logistic_class_050 == 0 & y_test == 0)
log_FN <- sum(logistic_class_050 == 0 & y_test == 1)
log_FP <- sum(logistic_class_050 == 1 & y_test == 0)
log_TP <- sum(logistic_class_050 == 1 & y_test == 1)

expected_confusion <- c(
  TN = 8781,
  FN = 1133,
  FP = 39,
  TP = 46
)

observed_confusion <- c(
  TN = log_TN,
  FN = log_FN,
  FP = log_FP,
  TP = log_TP
)

if (!identical(
  unname(observed_confusion),
  unname(expected_confusion)
)) {
  stop(
    paste0(
      "Frozen Logistic validation FAILED.\n",
      "Expected @0.50: ",
      paste(names(expected_confusion), expected_confusion, collapse = ", "),
      "\nObserved @0.50: ",
      paste(names(observed_confusion), observed_confusion, collapse = ", "),
      "\nDo not continue until the Study 1 artifact/population mismatch is resolved."
    )
  )
}

cat(
  "[OK] Logistic confusion matrix exactly matches proven thesis baseline: ",
  "TN=8781, FN=1133, FP=39, TP=46.\n",
  sep = ""
)


# ==============================================================================
# 7. LOAD FROZEN XGBOOST FINAL-TEST PREDICTIONS
# ==============================================================================

xgb_predictions <- read_csv(
  XGB_PREDICTIONS_FILE,
  show_col_types = FALSE
)

required_xgb_columns <- c(
  "LoanID",
  "ActualDefault",
  "RawXGBProbability",
  "PlattProbability",
  "SelectedProbabilityVariant",
  "SelectedProbability",
  "PredictedClass_050",
  "PredictedClass_ResearchThreshold"
)

missing_xgb_columns <- setdiff(
  required_xgb_columns,
  names(xgb_predictions)
)

if (length(missing_xgb_columns) > 0) {
  stop(
    paste0(
      "Frozen XGBoost prediction file is missing column(s): ",
      paste(missing_xgb_columns, collapse = ", ")
    )
  )
}

if (
  nrow(xgb_predictions) != 9999 ||
  n_distinct(xgb_predictions$LoanID) != 9999
) {
  stop(
    "Frozen XGBoost prediction file must contain exactly 9,999 unique LoanIDs."
  )
}

xgb_threshold_table <- read_csv(
  XGB_THRESHOLD_FILE,
  show_col_types = FALSE
)

if (!"Threshold" %in% names(xgb_threshold_table)) {
  stop(
    "Could not find Threshold column in frozen XGBoost threshold artifact."
  )
}

xgb_research_threshold <- as.numeric(
  xgb_threshold_table$Threshold[1]
)

if (
  !is.finite(xgb_research_threshold) ||
  xgb_research_threshold <= 0 ||
  xgb_research_threshold >= 1
) {
  stop("Frozen XGBoost research threshold is invalid.")
}

cat(
  sprintf(
    "[OK] Frozen XGBoost predictions loaded. Research threshold = %.3f\n",
    xgb_research_threshold
  )
)


# ==============================================================================
# 8. PAIR LOGISTIC AND XGBOOST BY LoanID
# ==============================================================================

logistic_scored <- tibble(
  LoanID = test_loan_ids,
  ActualDefault_Logistic = y_test,
  LogisticProbability = logistic_prob
)

paired <- logistic_scored %>%
  inner_join(
    xgb_predictions %>%
      transmute(
        LoanID = as.character(LoanID),
        ActualDefault_XGB = as.integer(ActualDefault),
        RawXGBProbability = as.numeric(RawXGBProbability),
        PlattProbability = as.numeric(PlattProbability),
        SelectedProbabilityVariant = as.character(
          SelectedProbabilityVariant
        ),
        XGBoostProbability = as.numeric(
          SelectedProbability
        ),
        XGBoostClass_050 = as.integer(
          PredictedClass_050
        ),
        XGBoostClass_ResearchThreshold = as.integer(
          PredictedClass_ResearchThreshold
        )
      ),
    by = "LoanID"
  )

if (nrow(paired) != 9999) {
  stop(
    paste0(
      "LoanID pairing failed. Expected 9,999 matched loans, found ",
      nrow(paired),
      "."
    )
  )
}

if (n_distinct(paired$LoanID) != 9999) {
  stop("Paired comparison contains duplicate LoanIDs.")
}

if (!all(
  paired$ActualDefault_Logistic ==
    paired$ActualDefault_XGB
)) {
  stop(
    "Target mismatch detected between Logistic and XGBoost paired rows."
  )
}

if (!setequal(
  paired$LoanID,
  test_loan_ids
)) {
  stop(
    "The frozen XGBoost holdout LoanIDs do not exactly match Study 1 holdout LoanIDs."
  )
}

paired <- paired %>%
  mutate(
    ActualDefault = ActualDefault_Logistic
  ) %>%
  select(
    LoanID,
    ActualDefault,
    LogisticProbability,
    XGBoostProbability,
    RawXGBProbability,
    PlattProbability,
    SelectedProbabilityVariant,
    XGBoostClass_050,
    XGBoostClass_ResearchThreshold
  )

write_csv(
  paired,
  file.path(TABLE_DIR, "paired_9999_model_predictions.csv")
)

cat("[OK] Exact LoanID pairing validated for all 9,999 holdout loans.\n")


# ==============================================================================
# 9. HELPER FUNCTIONS
# ==============================================================================

safe_probability <- function(p, eps = 1e-7) {
  pmin(
    pmax(as.numeric(p), eps),
    1 - eps
  )
}

binary_logloss <- function(y, p) {
  p <- safe_probability(p)

  -mean(
    y * log(p) +
      (1 - y) * log(1 - p)
  )
}

brier_score <- function(y, p) {
  mean(
    (as.numeric(p) - y)^2
  )
}

roc_auc_value <- function(y, p) {
  as.numeric(
    pROC::auc(
      pROC::roc(
        response = y,
        predictor = p,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
    )
  )
}

pr_auc_value <- function(y, p) {

  pr <- PRROC::pr.curve(
    scores.class0 = as.numeric(
      p[y == 1]
    ),
    scores.class1 = as.numeric(
      p[y == 0]
    ),
    curve = FALSE
  )

  as.numeric(
    pr$auc.integral
  )
}

classification_metrics <- function(
    y,
    p,
    threshold,
    model_name,
    threshold_label
) {

  pred <- ifelse(
    p >= threshold,
    1L,
    0L
  )

  TP <- sum(pred == 1 & y == 1)
  TN <- sum(pred == 0 & y == 0)
  FP <- sum(pred == 1 & y == 0)
  FN <- sum(pred == 0 & y == 1)

  accuracy <- (TP + TN) / length(y)

  recall <- if ((TP + FN) > 0) {
    TP / (TP + FN)
  } else {
    NA_real_
  }

  specificity <- if ((TN + FP) > 0) {
    TN / (TN + FP)
  } else {
    NA_real_
  }

  precision <- if ((TP + FP) > 0) {
    TP / (TP + FP)
  } else {
    NA_real_
  }

  f1 <- if (
    !is.na(precision) &&
      !is.na(recall) &&
      (precision + recall) > 0
  ) {
    2 * precision * recall /
      (precision + recall)
  } else {
    NA_real_
  }

  balanced_accuracy <- mean(
    c(
      recall,
      specificity
    ),
    na.rm = TRUE
  )

  tibble(
    Model = model_name,
    ThresholdLabel = threshold_label,
    Threshold = threshold,
    TN = TN,
    FN = FN,
    FP = FP,
    TP = TP,
    Accuracy = accuracy,
    Recall = recall,
    Specificity = specificity,
    Precision = precision,
    F1 = f1,
    BalancedAccuracy = balanced_accuracy,
    RiskFlagRate = mean(pred == 1)
  )
}

probability_metrics <- function(
    y,
    p,
    model_name
) {

  tibble(
    Model = model_name,
    ROC_AUC = roc_auc_value(y, p),
    PR_AUC = pr_auc_value(y, p),
    Brier = brier_score(y, p),
    LogLoss = binary_logloss(y, p)
  )
}


# ==============================================================================
# 10. PRIMARY PROBABILITY-LEVEL PERFORMANCE COMPARISON
# ==============================================================================

y <- paired$ActualDefault

log_p <- safe_probability(
  paired$LogisticProbability
)

xgb_p <- safe_probability(
  paired$XGBoostProbability
)

probability_comparison <- bind_rows(
  probability_metrics(
    y,
    log_p,
    "Full Logistic Regression"
  ),
  probability_metrics(
    y,
    xgb_p,
    "XGBoost Challenger"
  )
)

write_csv(
  probability_comparison,
  file.path(TABLE_DIR, "probability_metric_comparison.csv")
)

probability_differences <- tibble(
  Metric = c(
    "ROC-AUC",
    "PR-AUC",
    "Brier",
    "LogLoss"
  ),
  Logistic = c(
    probability_comparison$ROC_AUC[
      probability_comparison$Model ==
        "Full Logistic Regression"
    ],
    probability_comparison$PR_AUC[
      probability_comparison$Model ==
        "Full Logistic Regression"
    ],
    probability_comparison$Brier[
      probability_comparison$Model ==
        "Full Logistic Regression"
    ],
    probability_comparison$LogLoss[
      probability_comparison$Model ==
        "Full Logistic Regression"
    ]
  ),
  XGBoost = c(
    probability_comparison$ROC_AUC[
      probability_comparison$Model ==
        "XGBoost Challenger"
    ],
    probability_comparison$PR_AUC[
      probability_comparison$Model ==
        "XGBoost Challenger"
    ],
    probability_comparison$Brier[
      probability_comparison$Model ==
        "XGBoost Challenger"
    ],
    probability_comparison$LogLoss[
      probability_comparison$Model ==
        "XGBoost Challenger"
    ]
  )
) %>%
  mutate(
    XGB_minus_Logistic = XGBoost - Logistic,
    BetterDirection = c(
      "Higher",
      "Higher",
      "Lower",
      "Lower"
    )
  )

write_csv(
  probability_differences,
  file.path(TABLE_DIR, "probability_metric_differences.csv")
)


# ==============================================================================
# 11. PAIRED DELONG TEST FOR ROC-AUC
# ==============================================================================

roc_logistic <- pROC::roc(
  response = y,
  predictor = log_p,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_xgb <- pROC::roc(
  response = y,
  predictor = xgb_p,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

auc_logistic <- as.numeric(
  pROC::auc(roc_logistic)
)

auc_xgb <- as.numeric(
  pROC::auc(roc_xgb)
)

auc_ci_logistic <- as.numeric(
  pROC::ci.auc(
    roc_logistic,
    method = "delong"
  )
)

auc_ci_xgb <- as.numeric(
  pROC::ci.auc(
    roc_xgb,
    method = "delong"
  )
)

delong_test <- pROC::roc.test(
  roc_logistic,
  roc_xgb,
  paired = TRUE,
  method = "delong",
  alternative = "two.sided"
)

delong_results <- tibble(
  Logistic_ROC_AUC = auc_logistic,
  Logistic_CI_Lower = auc_ci_logistic[1],
  Logistic_CI_Upper = auc_ci_logistic[3],
  XGBoost_ROC_AUC = auc_xgb,
  XGBoost_CI_Lower = auc_ci_xgb[1],
  XGBoost_CI_Upper = auc_ci_xgb[3],
  Delta_AUC_XGB_minus_Logistic =
    auc_xgb - auc_logistic,
  DeLong_Z = as.numeric(
    delong_test$statistic
  ),
  P_Value = as.numeric(
    delong_test$p.value
  ),
  Significant_At_0_05 =
    as.numeric(delong_test$p.value) < ALPHA
)

write_csv(
  delong_results,
  file.path(TABLE_DIR, "paired_delong_roc_auc_test.csv")
)


# ==============================================================================
# 12. PAIRED STRATIFIED BOOTSTRAP
#     PR-AUC, BRIER, LOG LOSS DIFFERENCES
# ==============================================================================

positive_idx <- which(y == 1)
negative_idx <- which(y == 0)

n_pos <- length(positive_idx)
n_neg <- length(negative_idx)

bootstrap_results <- tibble(
  Iteration = seq_len(N_BOOT),
  Delta_PR_AUC = NA_real_,
  Delta_Brier = NA_real_,
  Delta_LogLoss = NA_real_
)

set.seed(BOOT_SEED)

cat(
  sprintf(
    "[INFO] Running %d paired stratified bootstrap replicates...\n",
    N_BOOT
  )
)

for (b in seq_len(N_BOOT)) {

  boot_pos <- sample(
    positive_idx,
    size = n_pos,
    replace = TRUE
  )

  boot_neg <- sample(
    negative_idx,
    size = n_neg,
    replace = TRUE
  )

  idx <- c(
    boot_pos,
    boot_neg
  )

  y_b <- y[idx]
  log_b <- log_p[idx]
  xgb_b <- xgb_p[idx]

  pr_log_b <- pr_auc_value(
    y_b,
    log_b
  )

  pr_xgb_b <- pr_auc_value(
    y_b,
    xgb_b
  )

  bootstrap_results$Delta_PR_AUC[b] <-
    pr_xgb_b - pr_log_b

  bootstrap_results$Delta_Brier[b] <-
    brier_score(y_b, xgb_b) -
    brier_score(y_b, log_b)

  bootstrap_results$Delta_LogLoss[b] <-
    binary_logloss(y_b, xgb_b) -
    binary_logloss(y_b, log_b)

  if (
    b %% 250L == 0L ||
      b == N_BOOT
  ) {
    cat(
      sprintf(
        "[BOOTSTRAP] %d / %d complete\n",
        b,
        N_BOOT
      )
    )
  }
}

write_csv(
  bootstrap_results,
  file.path(TABLE_DIR, "paired_bootstrap_replicates.csv")
)

bootstrap_ci <- function(x) {
  as.numeric(
    quantile(
      x,
      probs = c(
        ALPHA / 2,
        1 - ALPHA / 2
      ),
      na.rm = TRUE,
      type = 6
    )
  )
}

bootstrap_empirical_p <- function(x) {

  p_lower <- mean(
    x <= 0,
    na.rm = TRUE
  )

  p_upper <- mean(
    x >= 0,
    na.rm = TRUE
  )

  min(
    1,
    2 * min(
      p_lower,
      p_upper
    )
  )
}

pr_diff_observed <-
  pr_auc_value(y, xgb_p) -
  pr_auc_value(y, log_p)

brier_diff_observed <-
  brier_score(y, xgb_p) -
  brier_score(y, log_p)

logloss_diff_observed <-
  binary_logloss(y, xgb_p) -
  binary_logloss(y, log_p)

pr_ci <- bootstrap_ci(
  bootstrap_results$Delta_PR_AUC
)

brier_ci <- bootstrap_ci(
  bootstrap_results$Delta_Brier
)

logloss_ci <- bootstrap_ci(
  bootstrap_results$Delta_LogLoss
)

bootstrap_summary <- tibble(
  Metric = c(
    "PR-AUC",
    "Brier",
    "LogLoss"
  ),
  DifferenceDefinition = c(
    "XGBoost - Logistic; higher favors XGBoost",
    "XGBoost - Logistic; lower favors XGBoost",
    "XGBoost - Logistic; lower favors XGBoost"
  ),
  ObservedDifference = c(
    pr_diff_observed,
    brier_diff_observed,
    logloss_diff_observed
  ),
  CI_Lower = c(
    pr_ci[1],
    brier_ci[1],
    logloss_ci[1]
  ),
  CI_Upper = c(
    pr_ci[2],
    brier_ci[2],
    logloss_ci[2]
  ),
  EmpiricalTwoSidedP = c(
    bootstrap_empirical_p(
      bootstrap_results$Delta_PR_AUC
    ),
    bootstrap_empirical_p(
      bootstrap_results$Delta_Brier
    ),
    bootstrap_empirical_p(
      bootstrap_results$Delta_LogLoss
    )
  )
)

write_csv(
  bootstrap_summary,
  file.path(TABLE_DIR, "paired_bootstrap_summary.csv")
)


# ==============================================================================
# 13. FAIR THRESHOLD COMPARISON AT THE SAME 0.50 THRESHOLD
# ==============================================================================

threshold_050_comparison <- bind_rows(
  classification_metrics(
    y = y,
    p = log_p,
    threshold = 0.50,
    model_name = "Full Logistic Regression",
    threshold_label = "Same conventional threshold"
  ),
  classification_metrics(
    y = y,
    p = xgb_p,
    threshold = 0.50,
    model_name = "XGBoost Challenger",
    threshold_label = "Same conventional threshold"
  )
)

write_csv(
  threshold_050_comparison,
  file.path(TABLE_DIR, "classification_comparison_at_050.csv")
)


# ==============================================================================
# 14. MCNEMAR TEST AT THE SAME 0.50 THRESHOLD
# ==============================================================================

log_class_050 <- ifelse(
  log_p >= 0.50,
  1L,
  0L
)

xgb_class_050 <- ifelse(
  xgb_p >= 0.50,
  1L,
  0L
)

log_correct <- log_class_050 == y
xgb_correct <- xgb_class_050 == y

correctness_table <- table(
  LogisticCorrect = log_correct,
  XGBoostCorrect = xgb_correct
)

mcnemar_result <- mcnemar.test(
  correctness_table,
  correct = TRUE
)

mcnemar_output <- tibble(
  ComparisonThreshold = 0.50,
  LogisticWrong_XGBWrong =
    sum(!log_correct & !xgb_correct),
  LogisticWrong_XGBCorrect =
    sum(!log_correct & xgb_correct),
  LogisticCorrect_XGBWrong =
    sum(log_correct & !xgb_correct),
  LogisticCorrect_XGBCorrect =
    sum(log_correct & xgb_correct),
  ChiSquared = as.numeric(
    mcnemar_result$statistic
  ),
  P_Value = as.numeric(
    mcnemar_result$p.value
  ),
  Significant_At_0_05 =
    as.numeric(mcnemar_result$p.value) < ALPHA
)

write_csv(
  mcnemar_output,
  file.path(TABLE_DIR, "mcnemar_same_threshold_050.csv")
)


# ==============================================================================
# 15. XGBOOST RESEARCH-THRESHOLD RESULT — DESCRIPTIVE ONLY
# ==============================================================================

xgb_research_threshold_result <- classification_metrics(
  y = y,
  p = xgb_p,
  threshold = xgb_research_threshold,
  model_name = "XGBoost Challenger",
  threshold_label =
    "Development-OOF-selected XGBoost research threshold"
) %>%
  mutate(
    Interpretation =
      paste(
        "Descriptive XGBoost operating point only.",
        "Do not compare directly with Logistic @0.50",
        "as evidence of algorithmic superiority."
      )
  )

write_csv(
  xgb_research_threshold_result,
  file.path(
    TABLE_DIR,
    "xgb_research_threshold_descriptive_only.csv"
  )
)


# ==============================================================================
# 16. ROC FIGURE
# ==============================================================================

roc_plot_data <- bind_rows(
  tibble(
    Model = "Full Logistic Regression",
    FPR = 1 - roc_logistic$specificities,
    TPR = roc_logistic$sensitivities
  ),
  tibble(
    Model = "XGBoost Challenger",
    FPR = 1 - roc_xgb$specificities,
    TPR = roc_xgb$sensitivities
  )
)

p_roc <- ggplot(
  roc_plot_data,
  aes(
    x = FPR,
    y = TPR,
    linetype = Model
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = 3
  ) +
  geom_line(
    linewidth = 0.9
  ) +
  coord_equal() +
  labs(
    title = "Paired Holdout ROC Curves",
    subtitle = paste0(
      "Same 9,999 loans | Logistic AUC=",
      sprintf("%.4f", auc_logistic),
      " | XGBoost AUC=",
      sprintf("%.4f", auc_xgb)
    ),
    x = "False positive rate",
    y = "True positive rate"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(
    FIGURE_DIR,
    "paired_holdout_roc_curves.png"
  ),
  p_roc,
  width = 8,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 17. PROBABILITY-METRIC COMPARISON FIGURE
# ==============================================================================

discrimination_plot_data <- probability_comparison %>%
  select(
    Model,
    ROC_AUC,
    PR_AUC
  ) %>%
  pivot_longer(
    cols = c(
      ROC_AUC,
      PR_AUC
    ),
    names_to = "Metric",
    values_to = "Value"
  )

p_discrimination <- ggplot(
  discrimination_plot_data,
  aes(
    x = Metric,
    y = Value,
    fill = Model
  )
) +
  geom_col(
    position = "dodge"
  ) +
  labs(
    title = "Holdout Discrimination Comparison",
    subtitle = "Same 9,999 loans",
    x = NULL,
    y = "Area under curve"
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = "bottom"
  )

ggsave(
  file.path(
    FIGURE_DIR,
    "roc_pr_auc_comparison.png"
  ),
  p_discrimination,
  width = 8,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 18. BUILD A CONSERVATIVE EVIDENCE CONCLUSION
# ==============================================================================

delong_p <- delong_results$P_Value

pr_ci_lower <- bootstrap_summary %>%
  filter(Metric == "PR-AUC") %>%
  pull(CI_Lower)

pr_ci_upper <- bootstrap_summary %>%
  filter(Metric == "PR-AUC") %>%
  pull(CI_Upper)

auc_delta <- auc_xgb - auc_logistic
pr_delta <- pr_diff_observed

auc_evidence <- case_when(
  delong_p < ALPHA & auc_delta > 0 ~
    "Statistically significant ROC-AUC advantage for XGBoost",
  delong_p < ALPHA & auc_delta < 0 ~
    "Statistically significant ROC-AUC advantage for Logistic",
  TRUE ~
    "No statistically significant ROC-AUC difference"
)

pr_evidence <- case_when(
  pr_ci_lower > 0 ~
    "Paired bootstrap supports higher PR-AUC for XGBoost",
  pr_ci_upper < 0 ~
    "Paired bootstrap supports higher PR-AUC for Logistic",
  TRUE ~
    "Paired bootstrap PR-AUC CI includes zero"
)

overall_evidence <- case_when(
  delong_p < ALPHA &
    auc_delta > 0 &
    pr_ci_lower > 0 ~
    paste(
      "Both ROC-AUC and PR-AUC analyses support",
      "a holdout discrimination advantage for XGBoost."
    ),
  delong_p < ALPHA &
    auc_delta < 0 &
    pr_ci_upper < 0 ~
    paste(
      "Both ROC-AUC and PR-AUC analyses support",
      "a holdout discrimination advantage for Logistic Regression."
    ),
  TRUE ~
    paste(
      "Evidence of model superiority is mixed or insufficient.",
      "Report metric differences and uncertainty rather than",
      "claiming a definitive winner."
    )
)

evidence_conclusion <- tibble(
  Item = c(
    "ROC-AUC evidence",
    "PR-AUC evidence",
    "Overall conclusion"
  ),
  Conclusion = c(
    auc_evidence,
    pr_evidence,
    overall_evidence
  )
)

write_csv(
  evidence_conclusion,
  file.path(TABLE_DIR, "comparison_evidence_conclusion.csv")
)


# ==============================================================================
# 19. FINAL RESEARCH SUMMARY TABLE
# ==============================================================================

log_prob_row <- probability_comparison %>%
  filter(Model == "Full Logistic Regression")

xgb_prob_row <- probability_comparison %>%
  filter(Model == "XGBoost Challenger")

log_050_row <- threshold_050_comparison %>%
  filter(Model == "Full Logistic Regression")

xgb_050_row <- threshold_050_comparison %>%
  filter(Model == "XGBoost Challenger")

final_comparison_summary <- tibble(
  Metric = c(
    "ROC-AUC",
    "PR-AUC",
    "Brier score",
    "Log loss",
    "Recall @ 0.50",
    "Specificity @ 0.50",
    "Precision @ 0.50",
    "F1 @ 0.50",
    "Balanced accuracy @ 0.50"
  ),
  Logistic = c(
    log_prob_row$ROC_AUC,
    log_prob_row$PR_AUC,
    log_prob_row$Brier,
    log_prob_row$LogLoss,
    log_050_row$Recall,
    log_050_row$Specificity,
    log_050_row$Precision,
    log_050_row$F1,
    log_050_row$BalancedAccuracy
  ),
  XGBoost = c(
    xgb_prob_row$ROC_AUC,
    xgb_prob_row$PR_AUC,
    xgb_prob_row$Brier,
    xgb_prob_row$LogLoss,
    xgb_050_row$Recall,
    xgb_050_row$Specificity,
    xgb_050_row$Precision,
    xgb_050_row$F1,
    xgb_050_row$BalancedAccuracy
  )
) %>%
  mutate(
    XGB_minus_Logistic =
      XGBoost - Logistic
  )

write_csv(
  final_comparison_summary,
  file.path(TABLE_DIR, "final_logistic_vs_xgboost_summary.csv")
)


# ==============================================================================
# 20. AUDIT MANIFEST
# ==============================================================================

audit_manifest <- tibble(
  Item = c(
    "Dataset",
    "Dataset MD5",
    "Frozen Logistic model",
    "XGBoost source file",
    "Frozen XGBoost predictions",
    "Frozen XGBoost research threshold",
    "Paired holdout N",
    "Logistic retrained?",
    "XGBoost retrained?",
    "Primary Logistic model",
    "Bootstrap replicates",
    "Primary ROC test",
    "Primary PR-AUC uncertainty method"
  ),
  Value = c(
    DATA_FILE,
    EXPECTED_MD5,
    LOGISTIC_MODEL_FILE,
    ifelse(
      file.exists(XGB_SOURCE_FILE),
      XGB_SOURCE_FILE,
      paste0(
        XGB_SOURCE_FILE,
        " [not found in current working directory]"
      )
    ),
    XGB_PREDICTIONS_FILE,
    sprintf(
      "%.3f",
      xgb_research_threshold
    ),
    as.character(nrow(paired)),
    "NO",
    "NO",
    "Full Logistic Regression",
    as.character(N_BOOT),
    "Paired DeLong test",
    "Paired stratified bootstrap 95% CI"
  )
)

write_csv(
  audit_manifest,
  file.path(META_DIR, "comparison_audit_manifest.csv")
)


# ==============================================================================
# 21. COMPLETION MESSAGE
# ==============================================================================

cat("\n")
cat("======================================================================\n")
cat("LOGISTIC vs XGBOOST COMPARISON COMPLETE\n")
cat("======================================================================\n")
cat(sprintf(
  "Dataset MD5                         : %s\n",
  EXPECTED_MD5
))
cat(sprintf(
  "Paired holdout loans                 : %d\n",
  nrow(paired)
))
cat("Frozen Logistic model                : FULL Logistic Regression\n")
cat("Frozen XGBoost source                 : loan_default_xgb.R\n")
cat("Retraining performed                  : NONE\n")
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "Logistic ROC-AUC                      : %.4f\n",
  log_prob_row$ROC_AUC
))
cat(sprintf(
  "XGBoost ROC-AUC                       : %.4f\n",
  xgb_prob_row$ROC_AUC
))
cat(sprintf(
  "Delta ROC-AUC (XGB - Logistic)        : %+.4f\n",
  auc_delta
))
cat(sprintf(
  "Paired DeLong p-value                 : %.6f\n",
  delong_p
))
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "Logistic PR-AUC                       : %.4f\n",
  log_prob_row$PR_AUC
))
cat(sprintf(
  "XGBoost PR-AUC                        : %.4f\n",
  xgb_prob_row$PR_AUC
))
cat(sprintf(
  "Delta PR-AUC (XGB - Logistic)         : %+.4f\n",
  pr_delta
))
cat(sprintf(
  "PR-AUC bootstrap 95%% CI               : [%.4f, %.4f]\n",
  pr_ci_lower,
  pr_ci_upper
))
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "Logistic Brier                        : %.4f\n",
  log_prob_row$Brier
))
cat(sprintf(
  "XGBoost Brier                         : %.4f\n",
  xgb_prob_row$Brier
))
cat(sprintf(
  "Logistic Log Loss                     : %.4f\n",
  log_prob_row$LogLoss
))
cat(sprintf(
  "XGBoost Log Loss                      : %.4f\n",
  xgb_prob_row$LogLoss
))
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "Logistic Recall @0.50                 : %.4f\n",
  log_050_row$Recall
))
cat(sprintf(
  "XGBoost Recall @0.50                  : %.4f\n",
  xgb_050_row$Recall
))
cat(sprintf(
  "Logistic F1 @0.50                     : %.4f\n",
  log_050_row$F1
))
cat(sprintf(
  "XGBoost F1 @0.50                      : %.4f\n",
  xgb_050_row$F1
))
cat(sprintf(
  "McNemar p-value @0.50                 : %.6f\n",
  mcnemar_output$P_Value
))
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "XGBoost research threshold            : %.3f\n",
  xgb_research_threshold
))
cat(sprintf(
  "XGBoost Recall @ research threshold   : %.4f\n",
  xgb_research_threshold_result$Recall
))
cat(sprintf(
  "XGBoost Precision @ research threshold: %.4f\n",
  xgb_research_threshold_result$Precision
))
cat(sprintf(
  "XGBoost F1 @ research threshold       : %.4f\n",
  xgb_research_threshold_result$F1
))
cat(
  "NOTE: XGBoost research-threshold metrics are descriptive only;\n",
  "      they are NOT directly compared against Logistic @0.50.\n",
  sep = ""
)
cat("----------------------------------------------------------------------\n")
cat("ROC evidence                          : ", auc_evidence, "\n", sep = "")
cat("PR-AUC evidence                       : ", pr_evidence, "\n", sep = "")
cat("Overall evidence                      : ", overall_evidence, "\n", sep = "")
cat("----------------------------------------------------------------------\n")
cat(sprintf(
  "Outputs                               : %s\n",
  normalizePath(OUTPUT_DIR)
))
cat("======================================================================\n")
