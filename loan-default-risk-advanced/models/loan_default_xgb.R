# ==============================================================================
# STUDY 2: ADVANCED XGBOOST CHALLENGER MODEL
# Repository folder: loan-default-risk-advanced/
# File: loan_default_xgb_challenger_v2.R
#
# PURPOSE
# -------
# Build an advanced XGBoost binary-classification challenger without modifying
# the proven Logistic Regression baseline.
#
# Key design rules:
#   1) Same Loan_default.csv snapshot as Study 1.
#   2) Same deterministic 50,000-row sample (set.seed(1) + dplyr::sample_n).
#   3) Same factor-based stratified 80/20 split (set.seed(123)).
#   4) 40,001 development rows / 9,999 locked final-test rows.
#   5) All tuning, calibration, threshold selection, and SHAP development use
#      DEVELOPMENT data only.
#   6) Final test is evaluated only after the challenger specification is frozen.
#   7) XGBoost selection is driven primarily by PR-AUC because Default=1 is rare.
#   8) OOT validation is NOT claimed because the source dataset has no true
#      temporal/origination/performance date.
#
# IMPORTANT
# ---------
# This file does NOT alter the Logistic Regression study and does NOT combine
# Logistic + XGBoost. A later model_comparison.R can compare the frozen Logistic
# baseline with the keyed final-test predictions produced here.
# ==============================================================================


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

DATA_FILE <- "Loan_default.csv"

EXPECTED_MD5 <- "5f5a3364753b86ef3472cba30af224ce"

OUTPUT_DIR <- "xgboost_study2_outputs"
TABLE_DIR  <- file.path(OUTPUT_DIR, "tables")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
META_DIR   <- file.path(OUTPUT_DIR, "metadata")
MODEL_DIR  <- "models"

# QUICK_TEST = TRUE is useful only to verify that the pipeline runs.
# For the final research run, keep QUICK_TEST = FALSE.
QUICK_TEST <- FALSE

N_TUNING_CANDIDATES <- if (QUICK_TEST) 6L else 18L
MAX_NROUNDS         <- if (QUICK_TEST) 250L else 800L
EARLY_STOPPING      <- if (QUICK_TEST) 25L else 50L

N_FOLDS <- 5L

# Threshold rule used for the research challenger.
# This is NOT a production underwriting cutoff.
THRESHOLD_GRID <- seq(0.01, 0.99, by = 0.005)

# Native SHAP is calculated on a reproducible development subset to keep
# compute/memory reasonable.
SHAP_SAMPLE_N <- if (QUICK_TEST) 1500L else 5000L

# Reproducibility seeds
SEED_SAMPLE    <- 1L
SEED_SPLIT     <- 123L
SEED_FOLDS     <- 456L
SEED_GRID      <- 789L
SEED_XGB       <- 42L
SEED_SHAP      <- 999L

# CPU threads. Change manually if desired.
NTHREAD <- max(1L, parallel::detectCores(logical = TRUE) - 1L)


# ==============================================================================
# 1. REQUIRED PACKAGES AND OUTPUT FOLDERS
# ==============================================================================

required_packages <- c(
  "tidyverse",
  "caret",
  "xgboost",
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

install.packages(
  c("xgboost", "PRROC"),
  repos = "https://cloud.r-project.org",
  type = "binary"
)

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(xgboost)
  library(pROC)
  library(PRROC)
})

dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(TABLE_DIR, showWarnings = FALSE)
dir.create(FIGURE_DIR, showWarnings = FALSE)
dir.create(META_DIR, showWarnings = FALSE)
dir.create(MODEL_DIR, showWarnings = FALSE)

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
# 2. DATASET FINGERPRINT AND STRUCTURAL CHECKS
# ==============================================================================

if (!file.exists(DATA_FILE)) {
  stop(
    paste0(
      "CRITICAL ERROR: ", DATA_FILE, " not found.\n",
      "Place this R file in the same Study 2 folder as Loan_default.csv."
    )
  )
}

observed_md5 <- unname(tools::md5sum(DATA_FILE))

if (!identical(observed_md5, EXPECTED_MD5)) {
  stop(
    paste0(
      "Dataset fingerprint does not match the proven Study 1 snapshot.\n",
      "Expected MD5: ", EXPECTED_MD5, "\n",
      "Observed MD5: ", observed_md5, "\n",
      "Use the exact same Loan_default.csv before continuing."
    )
  )
}

writeLines(
  c(
    paste0("Data file: ", DATA_FILE),
    paste0("Observed MD5: ", observed_md5),
    paste0("Expected MD5: ", EXPECTED_MD5),
    paste0("Run timestamp: ", Sys.time()),
    paste0("QUICK_TEST: ", QUICK_TEST)
  ),
  con = file.path(META_DIR, "data_and_run_fingerprint.txt")
)

# Use read_csv to match Study 1's data-loading approach.
raw_data <- read_csv(DATA_FILE, show_col_types = FALSE)

stopifnot(
  nrow(raw_data) == 255347,
  ncol(raw_data) == 18,
  sum(is.na(raw_data)) == 0,
  sum(duplicated(raw_data)) == 0,
  sum(raw_data$Default == 1) == 29653,
  sum(raw_data$Default == 0) == 225694
)

cat("[OK] Dataset fingerprint and structural checks passed.\n")


# ==============================================================================
# 3. EXACT STUDY-1-COMPATIBLE 50,000-ROW SAMPLE
# ==============================================================================

set.seed(SEED_SAMPLE)

# IMPORTANT:
# Use the same operation as the proven Logistic reproducibility file.
sample_df <- dplyr::sample_n(raw_data, size = 50000)

rm(raw_data)
gc(verbose = FALSE)

stopifnot(nrow(sample_df) == 50000)

# Audit hashes for the sampled population.
sample_loan_ids <- as.character(sample_df$LoanID)

write_csv(
  tibble(LoanID = sample_loan_ids),
  file.path(META_DIR, "study2_sample_50000_loanids.csv")
)

sample_id_hash <- tools::md5sum(
  file.path(META_DIR, "study2_sample_50000_loanids.csv")
)

cat("[OK] Exact Study-1-style 50,000-row sampling step completed.\n")


# ==============================================================================
# 4. EXACT FACTOR-BASED STRATIFIED 80/20 PARTITION
# ==============================================================================

# Match Study 1: factorize character columns and make Default a factor
# BEFORE createDataPartition().
data_clean <- sample_df %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(Default = factor(Default, levels = c(0, 1)))

rm(sample_df)
gc(verbose = FALSE)

stopifnot(identical(levels(data_clean$Default), c("0", "1")))

set.seed(SEED_SPLIT)

train_idx <- createDataPartition(
  data_clean$Default,
  p = 0.80,
  list = FALSE
)

dev_data  <- data_clean[train_idx, ]
test_data <- data_clean[-train_idx, ]

# These are the proven Study 1 partition sizes.
stopifnot(
  nrow(dev_data) == 40001,
  nrow(test_data) == 9999
)

# Save IDs so a future Logistic-vs-XGBoost comparison can prove paired rows.
write_csv(
  tibble(RowOrder = seq_len(nrow(dev_data)),
         LoanID = as.character(dev_data$LoanID)),
  file.path(META_DIR, "development_loanids_40001.csv")
)

write_csv(
  tibble(RowOrder = seq_len(nrow(test_data)),
         LoanID = as.character(test_data$LoanID)),
  file.path(META_DIR, "locked_test_loanids_9999.csv")
)

split_summary <- tibble(
  Partition = c("Development", "Locked final test"),
  N = c(nrow(dev_data), nrow(test_data)),
  Defaults = c(
    sum(dev_data$Default == "1"),
    sum(test_data$Default == "1")
  ),
  NonDefaults = c(
    sum(dev_data$Default == "0"),
    sum(test_data$Default == "0")
  )
) %>%
  mutate(DefaultRate = Defaults / N)

write_csv(
  split_summary,
  file.path(TABLE_DIR, "study2_partition_summary.csv")
)

cat(
  sprintf(
    "[OK] Study-1-compatible split confirmed: %d development / %d locked test.\n",
    nrow(dev_data),
    nrow(test_data)
  )
)


# ==============================================================================
# 5. DEVELOPMENT-ONLY PREPROCESSING
# ==============================================================================

# XGBoost labels must be numeric 0/1.
y_dev <- as.numeric(as.character(dev_data$Default))

stopifnot(all(y_dev %in% c(0, 1)))

drop_cols <- c("LoanID", "Default")

x_dev_raw <- dev_data %>%
  select(-all_of(drop_cols))

# Fit dummy schema on DEVELOPMENT data only.
# fullRank = FALSE retains all one-hot category columns, which is acceptable
# for tree models and makes attribution labels easier to interpret.
dummifier <- dummyVars(
  ~ .,
  data = x_dev_raw,
  fullRank = FALSE
)

x_dev_mat <- predict(
  dummifier,
  newdata = x_dev_raw
)

x_dev_mat <- as.matrix(x_dev_mat)
storage.mode(x_dev_mat) <- "double"

if (anyNA(x_dev_mat)) {
  stop("Preprocessing produced NA values in the development matrix.")
}

feature_names <- colnames(x_dev_mat)

write_csv(
  tibble(Feature = feature_names),
  file.path(META_DIR, "xgboost_feature_schema.csv")
)

saveRDS(
  dummifier,
  file.path(MODEL_DIR, "xgb_preprocessor.rds")
)

neg_count <- sum(y_dev == 0)
pos_count <- sum(y_dev == 1)
scale_pos_w <- neg_count / pos_count

imbalance_summary <- tibble(
  DevelopmentN = length(y_dev),
  Negatives = neg_count,
  Positives = pos_count,
  DefaultRate = pos_count / length(y_dev),
  CalculatedScalePosWeight = scale_pos_w
)

write_csv(
  imbalance_summary,
  file.path(TABLE_DIR, "development_class_imbalance.csv")
)

cat(
  sprintf(
    "[INFO] Development negatives=%d, positives=%d, scale_pos_weight=%.4f\n",
    neg_count,
    pos_count,
    scale_pos_w
  )
)

dtrain <- xgb.DMatrix(
  data = x_dev_mat,
  label = y_dev,
  nthread = NTHREAD
)


# ==============================================================================
# 6. STRATIFIED 5-FOLD CROSS-VALIDATION FOLDS
# ==============================================================================

set.seed(SEED_FOLDS)

# Factor outcome ensures classification-stratified folds.
cv_folds <- createFolds(
  factor(y_dev, levels = c(0, 1)),
  k = N_FOLDS,
  list = TRUE,
  returnTrain = FALSE
)

# Verify each observation is held out exactly once across the folds.
all_fold_indices <- sort(unlist(cv_folds, use.names = FALSE))

stopifnot(
  identical(all_fold_indices, seq_len(length(y_dev)))
)

fold_summary <- map_dfr(
  seq_along(cv_folds),
  function(i) {
    idx <- cv_folds[[i]]
    tibble(
      Fold = i,
      N = length(idx),
      Defaults = sum(y_dev[idx] == 1),
      NonDefaults = sum(y_dev[idx] == 0),
      DefaultRate = mean(y_dev[idx] == 1)
    )
  }
)

write_csv(
  fold_summary,
  file.path(TABLE_DIR, "cv_fold_summary.csv")
)

cat("[OK] Stratified 5-fold CV structure created from development data only.\n")


# ==============================================================================
# 7. HELPER FUNCTIONS
# ==============================================================================

safe_probability <- function(p, eps = 1e-7) {
  pmin(pmax(as.numeric(p), eps), 1 - eps)
}

safe_logit <- function(p, eps = 1e-7) {
  qlogis(safe_probability(p, eps))
}

binary_logloss <- function(y, p) {
  p <- safe_probability(p)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

brier_score <- function(y, p) {
  mean((as.numeric(p) - y)^2)
}

roc_auc_value <- function(y, p) {
  if (length(unique(y)) < 2) return(NA_real_)
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
  if (length(unique(y)) < 2) return(NA_real_)

  pr <- PRROC::pr.curve(
    scores.class0 = as.numeric(p[y == 1]),
    scores.class1 = as.numeric(p[y == 0]),
    curve = FALSE
  )

  as.numeric(pr$auc.integral)
}

threshold_metrics <- function(y, p, threshold) {

  pred <- ifelse(p >= threshold, 1L, 0L)

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
    2 * precision * recall / (precision + recall)
  } else {
    NA_real_
  }

  balanced_accuracy <- mean(
    c(recall, specificity),
    na.rm = TRUE
  )

  fpr <- if ((FP + TN) > 0) {
    FP / (FP + TN)
  } else {
    NA_real_
  }

  fnr <- if ((FN + TP) > 0) {
    FN / (FN + TP)
  } else {
    NA_real_
  }

  risk_flag_rate <- mean(pred == 1)

  tibble(
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
    FPR = fpr,
    FNR = fnr,
    RiskFlagRate = risk_flag_rate
  )
}

probability_metrics <- function(y, p, label) {
  tibble(
    ProbabilityVariant = label,
    ROC_AUC = roc_auc_value(y, p),
    PR_AUC = pr_auc_value(y, p),
    Brier = brier_score(y, p),
    LogLoss = binary_logloss(y, p)
  )
}

calibration_table <- function(y, p, variant, n_bins = 10L) {

  tibble(
    y = y,
    p = p
  ) %>%
    mutate(
      Bin = ntile(p, n_bins)
    ) %>%
    group_by(Bin) %>%
    summarise(
      N = n(),
      MeanPredicted = mean(p),
      ObservedRate = mean(y),
      .groups = "drop"
    ) %>%
    mutate(
      Variant = variant
    )
}

extract_best_iteration <- function(cv_object) {

  # Current XGBoost R API
  if (
    !is.null(cv_object$early_stop) &&
      !is.null(cv_object$early_stop$best_iteration)
  ) {
    return(as.integer(cv_object$early_stop$best_iteration))
  }

  # Compatibility with some older releases
  if (!is.null(cv_object$best_iteration)) {
    return(as.integer(cv_object$best_iteration))
  }

  # Last-resort fallback: best PR-AUC row in evaluation log.
  log_tbl <- as.data.frame(cv_object$evaluation_log)

  score_col <- grep(
    "^test_aucpr_mean$",
    names(log_tbl),
    value = TRUE
  )

  if (length(score_col) != 1) {
    stop("Could not determine CV best iteration.")
  }

  as.integer(which.max(log_tbl[[score_col]]))
}

extract_cv_predictions <- function(cv_object) {

  # Current XGBoost R API
  if (
    !is.null(cv_object$cv_predict) &&
      !is.null(cv_object$cv_predict$pred)
  ) {
    return(as.numeric(cv_object$cv_predict$pred))
  }

  # Compatibility with older releases
  if (!is.null(cv_object$pred)) {
    return(as.numeric(cv_object$pred))
  }

  stop(
    paste0(
      "Could not find out-of-fold predictions in xgb.cv result. ",
      "Check the installed xgboost version and prediction=TRUE."
    )
  )
}


# ==============================================================================
# 8. HYPERPARAMETER SEARCH SPACE
# ==============================================================================

# We use a deterministic, controlled random search instead of a huge exhaustive
# grid. This gives real hyperparameter tuning while keeping runtime practical.

anchor_candidates <- tibble(
  eta = c(0.05, 0.05, 0.03, 0.08),
  max_depth = c(6L, 6L, 4L, 3L),
  min_child_weight = c(3, 3, 5, 1),
  gamma = c(0, 0, 0, 0.5),
  subsample = c(0.8, 0.8, 0.9, 0.8),
  colsample_bytree = c(0.8, 0.8, 0.9, 0.8),
  lambda = c(1, 1, 5, 5),
  alpha = c(0, 0, 0.1, 0.1),
  scale_pos_weight = c(
    1,
    scale_pos_w,
    scale_pos_w,
    1
  )
)

full_grid <- expand_grid(
  eta = c(0.025, 0.05, 0.08, 0.12),
  max_depth = c(3L, 4L, 6L, 8L),
  min_child_weight = c(1, 3, 6),
  gamma = c(0, 0.25, 0.75),
  subsample = c(0.75, 0.9, 1.0),
  colsample_bytree = c(0.75, 0.9, 1.0),
  lambda = c(1, 5, 10),
  alpha = c(0, 0.1, 0.5),
  scale_pos_weight = c(1, scale_pos_w)
)

set.seed(SEED_GRID)

n_extra <- max(
  0L,
  N_TUNING_CANDIDATES - nrow(anchor_candidates)
)

sampled_candidates <- full_grid %>%
  slice_sample(n = min(n_extra, nrow(full_grid)))

candidate_grid <- bind_rows(
  anchor_candidates,
  sampled_candidates
) %>%
  distinct() %>%
  slice_head(n = N_TUNING_CANDIDATES) %>%
  mutate(CandidateID = row_number()) %>%
  relocate(CandidateID)

write_csv(
  candidate_grid,
  file.path(TABLE_DIR, "xgb_hyperparameter_candidates.csv")
)

cat(
  sprintf(
    "[INFO] Hyperparameter search will evaluate %d candidates.\n",
    nrow(candidate_grid)
  )
)


# ==============================================================================
# 9. REAL 5-FOLD HYPERPARAMETER TUNING
#    PRIMARY SELECTION METRIC: PR-AUC (XGBoost aucpr)
# ==============================================================================

tuning_results_list <- vector(
  mode = "list",
  length = nrow(candidate_grid)
)

for (i in seq_len(nrow(candidate_grid))) {

  cand <- candidate_grid[i, ]

  cat(
    sprintf(
      "[TUNING %02d/%02d] depth=%d eta=%.3f child=%.1f gamma=%.2f weight=%.3f\n",
      i,
      nrow(candidate_grid),
      cand$max_depth,
      cand$eta,
      cand$min_child_weight,
      cand$gamma,
      cand$scale_pos_weight
    )
  )

  params_i <- list(
    booster = "gbtree",
    objective = "binary:logistic",

    # PR-AUC is primary for early stopping/model selection.
    eval_metric = "aucpr",

    tree_method = "hist",

    eta = cand$eta,
    max_depth = as.integer(cand$max_depth),
    min_child_weight = cand$min_child_weight,
    gamma = cand$gamma,
    subsample = cand$subsample,
    colsample_bytree = cand$colsample_bytree,
    lambda = cand$lambda,
    alpha = cand$alpha,
    scale_pos_weight = cand$scale_pos_weight,

    nthread = NTHREAD,
    seed = SEED_XGB
  )

  cv_i <- tryCatch(
    {
      set.seed(SEED_XGB + i)

      xgb.cv(
        params = params_i,
        data = dtrain,
        nrounds = MAX_NROUNDS,
        nfold = N_FOLDS,
        folds = cv_folds,
        early_stopping_rounds = EARLY_STOPPING,
        maximize = TRUE,
        prediction = FALSE,
        verbose = FALSE
      )
    },
    error = function(e) {
      message(
        "[WARNING] Candidate ",
        i,
        " failed: ",
        conditionMessage(e)
      )
      NULL
    }
  )

  if (is.null(cv_i)) {

    tuning_results_list[[i]] <- cand %>%
      mutate(
        BestIteration = NA_integer_,
        CV_PR_AUC = NA_real_,
        CV_PR_AUC_SD = NA_real_,
        Status = "FAILED"
      )

    next
  }

  best_iter_i <- extract_best_iteration(cv_i)

  eval_log_i <- as.data.frame(cv_i$evaluation_log)

  mean_col <- grep(
    "^test_aucpr_mean$",
    names(eval_log_i),
    value = TRUE
  )

  sd_col <- grep(
    "^test_aucpr_std$",
    names(eval_log_i),
    value = TRUE
  )

  if (length(mean_col) != 1) {
    stop("Could not locate test_aucpr_mean in xgb.cv evaluation log.")
  }

  best_pr_i <- eval_log_i[[mean_col]][best_iter_i]

  best_sd_i <- if (length(sd_col) == 1) {
    eval_log_i[[sd_col]][best_iter_i]
  } else {
    NA_real_
  }

  tuning_results_list[[i]] <- cand %>%
    mutate(
      BestIteration = best_iter_i,
      CV_PR_AUC = best_pr_i,
      CV_PR_AUC_SD = best_sd_i,
      Status = "OK"
    )

  rm(cv_i)
  gc(verbose = FALSE)
}

tuning_results <- bind_rows(tuning_results_list)

write_csv(
  tuning_results,
  file.path(TABLE_DIR, "xgb_tuning_results.csv")
)

valid_tuning_results <- tuning_results %>%
  filter(
    Status == "OK",
    !is.na(CV_PR_AUC)
  )

if (nrow(valid_tuning_results) == 0) {
  stop("All XGBoost tuning candidates failed.")
}

best_candidate <- valid_tuning_results %>%
  arrange(
    desc(CV_PR_AUC),
    CV_PR_AUC_SD,
    BestIteration
  ) %>%
  slice(1)

write_csv(
  best_candidate,
  file.path(TABLE_DIR, "xgb_best_hyperparameters.csv")
)

cat("\n[WINNER] Best XGBoost CV candidate:\n")
print(best_candidate)


# ==============================================================================
# 10. TUNING FIGURE
# ==============================================================================

tuning_plot_data <- valid_tuning_results %>%
  arrange(desc(CV_PR_AUC)) %>%
  mutate(
    Candidate = factor(
      paste0("C", CandidateID),
      levels = rev(paste0("C", CandidateID))
    )
  )

p_tuning <- ggplot(
  tuning_plot_data,
  aes(
    x = Candidate,
    y = CV_PR_AUC
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "XGBoost Hyperparameter Search",
    subtitle = "Primary cross-validation selection metric: PR-AUC",
    x = NULL,
    y = "Mean 5-fold CV PR-AUC"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "xgb_tuning_pr_auc.png"),
  p_tuning,
  width = 9,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 11. RE-RUN BEST CONFIGURATION FOR CLEAN OUT-OF-FOLD PREDICTIONS
# ==============================================================================

best_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "aucpr",
  tree_method = "hist",

  eta = best_candidate$eta,
  max_depth = as.integer(best_candidate$max_depth),
  min_child_weight = best_candidate$min_child_weight,
  gamma = best_candidate$gamma,
  subsample = best_candidate$subsample,
  colsample_bytree = best_candidate$colsample_bytree,
  lambda = best_candidate$lambda,
  alpha = best_candidate$alpha,
  scale_pos_weight = best_candidate$scale_pos_weight,

  nthread = NTHREAD,
  seed = SEED_XGB
)

best_iteration <- as.integer(best_candidate$BestIteration)

set.seed(SEED_XGB)

best_cv_oof <- xgb.cv(
  params = best_params,
  data = dtrain,
  nrounds = best_iteration,
  nfold = N_FOLDS,
  folds = cv_folds,
  prediction = TRUE,
  verbose = FALSE
)

oof_raw <- extract_cv_predictions(best_cv_oof)

stopifnot(
  length(oof_raw) == length(y_dev),
  all(is.finite(oof_raw))
)

oof_raw <- safe_probability(oof_raw)

write_csv(
  tibble(
    LoanID = as.character(dev_data$LoanID),
    ActualDefault = y_dev,
    OOF_RawProbability = oof_raw
  ),
  file.path(TABLE_DIR, "xgb_development_oof_raw_predictions.csv")
)

cat("[OK] Out-of-fold development predictions created without using locked test.\n")


# ==============================================================================
# 12. PLATT CALIBRATION USING OOF LOG-ODDS
# ==============================================================================

oof_logit <- safe_logit(oof_raw)

calib_model <- glm(
  y_dev ~ oof_logit,
  family = binomial(link = "logit")
)

oof_platt <- predict(
  calib_model,
  newdata = data.frame(oof_logit = oof_logit),
  type = "response"
)

oof_platt <- safe_probability(oof_platt)

oof_probability_metrics <- bind_rows(
  probability_metrics(
    y_dev,
    oof_raw,
    "Raw XGBoost OOF"
  ),
  probability_metrics(
    y_dev,
    oof_platt,
    "Platt-calibrated XGBoost OOF"
  )
)

write_csv(
  oof_probability_metrics,
  file.path(TABLE_DIR, "xgb_oof_probability_quality.csv")
)

raw_oof_brier <- oof_probability_metrics %>%
  filter(ProbabilityVariant == "Raw XGBoost OOF") %>%
  pull(Brier)

platt_oof_brier <- oof_probability_metrics %>%
  filter(ProbabilityVariant == "Platt-calibrated XGBoost OOF") %>%
  pull(Brier)

raw_oof_logloss <- oof_probability_metrics %>%
  filter(ProbabilityVariant == "Raw XGBoost OOF") %>%
  pull(LogLoss)

platt_oof_logloss <- oof_probability_metrics %>%
  filter(ProbabilityVariant == "Platt-calibrated XGBoost OOF") %>%
  pull(LogLoss)

# Transparent research rule:
# Use Platt only when both Brier and log loss improve on development OOF.
use_platt <- (
  platt_oof_brier < raw_oof_brier &&
    platt_oof_logloss < raw_oof_logloss
)

selected_probability_variant <- if (use_platt) {
  "Platt"
} else {
  "Raw"
}

oof_selected <- if (use_platt) {
  oof_platt
} else {
  oof_raw
}

calibration_decision <- tibble(
  RawOOF_Brier = raw_oof_brier,
  PlattOOF_Brier = platt_oof_brier,
  RawOOF_LogLoss = raw_oof_logloss,
  PlattOOF_LogLoss = platt_oof_logloss,
  UsePlatt = use_platt,
  SelectedProbabilityVariant = selected_probability_variant,
  DecisionRule = paste(
    "Use Platt only if BOTH OOF Brier score and OOF log loss",
    "are lower than raw XGBoost."
  )
)

write_csv(
  calibration_decision,
  file.path(TABLE_DIR, "xgb_calibration_decision.csv")
)

saveRDS(
  calib_model,
  file.path(MODEL_DIR, "xgb_platt_calibrator.rds")
)

cat(
  sprintf(
    "[INFO] Probability variant selected from development OOF: %s\n",
    selected_probability_variant
  )
)


# ==============================================================================
# 13. OOF CALIBRATION TABLE + FIGURE
# ==============================================================================

calibration_oof_table <- bind_rows(
  calibration_table(
    y_dev,
    oof_raw,
    "Raw"
  ),
  calibration_table(
    y_dev,
    oof_platt,
    "Platt"
  )
)

write_csv(
  calibration_oof_table,
  file.path(TABLE_DIR, "xgb_oof_calibration_deciles.csv")
)

p_calibration <- ggplot(
  calibration_oof_table,
  aes(
    x = MeanPredicted,
    y = ObservedRate,
    shape = Variant
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = 2
  ) +
  geom_point(size = 2.5) +
  geom_line(
    aes(group = Variant)
  ) +
  coord_equal() +
  labs(
    title = "Development OOF Calibration",
    subtitle = "Raw vs Platt-calibrated XGBoost probabilities",
    x = "Mean predicted probability",
    y = "Observed default rate"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "xgb_oof_calibration.png"),
  p_calibration,
  width = 7,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 14. DEVELOPMENT-ONLY THRESHOLD OPTIMIZATION
# ==============================================================================

threshold_table <- map_dfr(
  THRESHOLD_GRID,
  function(t) {
    threshold_metrics(
      y = y_dev,
      p = oof_selected,
      threshold = t
    )
  }
)

write_csv(
  threshold_table,
  file.path(TABLE_DIR, "xgb_oof_threshold_analysis.csv")
)

# Research challenger rule:
# maximize F1; if tied, prefer higher recall, then higher precision,
# then threshold closer to 0.50.
optimal_threshold_row <- threshold_table %>%
  filter(!is.na(F1)) %>%
  mutate(
    DistanceFrom050 = abs(Threshold - 0.50)
  ) %>%
  arrange(
    desc(F1),
    desc(Recall),
    desc(Precision),
    DistanceFrom050
  ) %>%
  slice(1)

opt_threshold <- optimal_threshold_row$Threshold

write_csv(
  optimal_threshold_row,
  file.path(TABLE_DIR, "xgb_selected_research_threshold.csv")
)

saveRDS(
  opt_threshold,
  file.path(MODEL_DIR, "xgb_research_threshold.rds")
)

cat(
  sprintf(
    "[INFO] Development OOF research threshold selected: %.3f\n",
    opt_threshold
  )
)

threshold_plot_data <- threshold_table %>%
  select(
    Threshold,
    Recall,
    Precision,
    F1,
    BalancedAccuracy
  ) %>%
  pivot_longer(
    cols = -Threshold,
    names_to = "Metric",
    values_to = "Value"
  )

p_threshold <- ggplot(
  threshold_plot_data,
  aes(
    x = Threshold,
    y = Value,
    linetype = Metric
  )
) +
  geom_line(linewidth = 0.8) +
  geom_vline(
    xintercept = opt_threshold,
    linetype = 2
  ) +
  labs(
    title = "Development OOF Threshold Analysis",
    subtitle = paste0(
      "Selected research threshold = ",
      sprintf("%.3f", opt_threshold)
    ),
    x = "Decision threshold",
    y = "Metric value"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "xgb_oof_threshold_analysis.png"),
  p_threshold,
  width = 9,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 15. FIT THE FROZEN FINAL XGBOOST CHALLENGER ON ALL DEVELOPMENT DATA
# ==============================================================================

cat("[INFO] Fitting frozen final XGBoost challenger on all development rows...\n")

set.seed(SEED_XGB)

final_xgb <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_iteration,
  verbose = 0
)

# Save both R-native and XGBoost-native model formats.
saveRDS(
  final_xgb,
  file.path(MODEL_DIR, "xgboost_final_model.rds")
)

xgb.save(
  final_xgb,
  file.path(MODEL_DIR, "xgboost_final_model.ubj")
)

model_config <- list(
  best_params = best_params,
  best_iteration = best_iteration,
  selected_probability_variant = selected_probability_variant,
  use_platt = use_platt,
  research_threshold = opt_threshold,
  feature_names = feature_names,
  expected_md5 = EXPECTED_MD5,
  sample_seed = SEED_SAMPLE,
  split_seed = SEED_SPLIT,
  folds_seed = SEED_FOLDS
)

saveRDS(
  model_config,
  file.path(MODEL_DIR, "xgb_model_config.rds")
)

cat("[OK] Final XGBoost challenger and deployment artifacts saved.\n")


# ==============================================================================
# 16. NATIVE XGBOOST SHAP EXPLAINABILITY
#     DEVELOPMENT DATA ONLY
# ==============================================================================

set.seed(SEED_SHAP)

shap_n <- min(
  SHAP_SAMPLE_N,
  nrow(x_dev_mat)
)

shap_idx <- sample(
  seq_len(nrow(x_dev_mat)),
  size = shap_n,
  replace = FALSE
)

x_shap <- x_dev_mat[shap_idx, , drop = FALSE]

# For xgb.Booster, predcontrib=TRUE returns exact TreeSHAP contributions
# on the raw margin/log-odds scale. Last column is the bias/intercept.
shap_contrib <- predict(
  final_xgb,
  xgb.DMatrix(
    x_shap,
    nthread = NTHREAD
  ),
  predcontrib = TRUE
)

shap_contrib <- as.matrix(shap_contrib)

if (ncol(shap_contrib) != ncol(x_shap) + 1) {
  stop("Unexpected SHAP contribution matrix dimensions.")
}

shap_values <- shap_contrib[
  ,
  seq_len(ncol(x_shap)),
  drop = FALSE
]

colnames(shap_values) <- colnames(x_shap)

shap_global <- tibble(
  Feature = colnames(shap_values),
  MeanAbsSHAP = colMeans(abs(shap_values))
) %>%
  arrange(desc(MeanAbsSHAP))

write_csv(
  shap_global,
  file.path(TABLE_DIR, "xgb_shap_global_importance.csv")
)

p_shap_global <- shap_global %>%
  slice_head(n = 20) %>%
  mutate(
    Feature = factor(
      Feature,
      levels = rev(Feature)
    )
  ) %>%
  ggplot(
    aes(
      x = Feature,
      y = MeanAbsSHAP
    )
  ) +
  geom_col() +
  coord_flip() +
  labs(
    title = "XGBoost Global SHAP Importance",
    subtitle = paste0(
      "Mean absolute SHAP contribution; development sample n=",
      shap_n
    ),
    x = NULL,
    y = "Mean |SHAP| on raw model-margin scale"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "xgb_shap_global_importance.png"),
  p_shap_global,
  width = 9,
  height = 7,
  dpi = 300
)

# Save a manageable row-level SHAP extract for later local explanation work.
top_shap_features <- shap_global %>%
  slice_head(n = min(15L, nrow(shap_global))) %>%
  pull(Feature)

shap_long_export <- as_tibble(
  shap_values[, top_shap_features, drop = FALSE]
) %>%
  mutate(
    SHAP_Row = seq_len(n()),
    LoanID = as.character(dev_data$LoanID[shap_idx])
  ) %>%
  relocate(
    SHAP_Row,
    LoanID
  )

write_csv(
  shap_long_export,
  file.path(TABLE_DIR, "xgb_shap_top_features_row_level.csv")
)

cat("[OK] Native TreeSHAP outputs generated from development data.\n")


# ==============================================================================
# 17. FINAL TEST UNLOCK
#     NO DEVELOPMENT/TUNING DECISIONS MAY BE CHANGED AFTER THIS POINT
# ==============================================================================

cat("\n")
cat("======================================================================\n")
cat("FINAL TEST UNLOCK\n")
cat("The XGBoost design, parameters, calibration rule, and threshold are frozen.\n")
cat(sprintf("Locked final test N = %d\n", nrow(test_data)))
cat("======================================================================\n")

# Create test objects only now.
y_test <- as.numeric(as.character(test_data$Default))

x_test_raw <- test_data %>%
  select(-all_of(drop_cols))

x_test_mat <- predict(
  dummifier,
  newdata = x_test_raw
)

x_test_mat <- as.matrix(x_test_mat)
storage.mode(x_test_mat) <- "double"

if (anyNA(x_test_mat)) {
  stop(
    paste0(
      "Preprocessing produced NA in locked test data. ",
      "This can indicate unseen categorical levels."
    )
  )
}

if (!identical(colnames(x_test_mat), feature_names)) {
  stop("Locked-test feature schema does not match development feature schema.")
}

dtest <- xgb.DMatrix(
  data = x_test_mat,
  label = y_test,
  nthread = NTHREAD
)

raw_test_prob <- predict(
  final_xgb,
  dtest
)

raw_test_prob <- safe_probability(raw_test_prob)

test_logit <- safe_logit(raw_test_prob)

platt_test_prob <- predict(
  calib_model,
  newdata = data.frame(oof_logit = test_logit),
  type = "response"
)

platt_test_prob <- safe_probability(platt_test_prob)

selected_test_prob <- if (use_platt) {
  platt_test_prob
} else {
  raw_test_prob
}

pred_test_050 <- ifelse(
  selected_test_prob >= 0.50,
  1L,
  0L
)

pred_test_opt <- ifelse(
  selected_test_prob >= opt_threshold,
  1L,
  0L
)

final_test_predictions <- tibble(
  RowOrder = seq_len(nrow(test_data)),
  LoanID = as.character(test_data$LoanID),
  ActualDefault = y_test,
  RawXGBProbability = raw_test_prob,
  PlattProbability = platt_test_prob,
  SelectedProbabilityVariant = selected_probability_variant,
  SelectedProbability = selected_test_prob,
  PredictedClass_050 = pred_test_050,
  PredictedClass_ResearchThreshold = pred_test_opt
)

write_csv(
  final_test_predictions,
  file.path(TABLE_DIR, "xgb_locked_test_keyed_predictions.csv")
)


# ==============================================================================
# 18. FINAL TEST PROBABILITY METRICS
# ==============================================================================

final_probability_metrics <- bind_rows(
  probability_metrics(
    y_test,
    raw_test_prob,
    "Raw XGBoost"
  ),
  probability_metrics(
    y_test,
    platt_test_prob,
    "Platt-calibrated XGBoost"
  ),
  probability_metrics(
    y_test,
    selected_test_prob,
    paste0(
      "Selected: ",
      selected_probability_variant
    )
  )
)

write_csv(
  final_probability_metrics,
  file.path(TABLE_DIR, "xgb_locked_test_probability_metrics.csv")
)


# ==============================================================================
# 19. FINAL TEST THRESHOLDED CLASSIFICATION METRICS
# ==============================================================================

final_threshold_metrics <- bind_rows(
  threshold_metrics(
    y_test,
    selected_test_prob,
    0.50
  ) %>%
    mutate(
      ThresholdLabel = "Conventional 0.50"
    ),

  threshold_metrics(
    y_test,
    selected_test_prob,
    opt_threshold
  ) %>%
    mutate(
      ThresholdLabel = "Development-selected research threshold"
    )
) %>%
  relocate(
    ThresholdLabel,
    Threshold
  )

write_csv(
  final_threshold_metrics,
  file.path(TABLE_DIR, "xgb_locked_test_classification_metrics.csv")
)


# ==============================================================================
# 20. FINAL TEST CALIBRATION DIAGNOSTIC
# ==============================================================================

final_calibration_table <- bind_rows(
  calibration_table(
    y_test,
    raw_test_prob,
    "Raw"
  ),
  calibration_table(
    y_test,
    platt_test_prob,
    "Platt"
  )
)

write_csv(
  final_calibration_table,
  file.path(TABLE_DIR, "xgb_locked_test_calibration_deciles.csv")
)

p_test_calibration <- ggplot(
  final_calibration_table,
  aes(
    x = MeanPredicted,
    y = ObservedRate,
    shape = Variant
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = 2
  ) +
  geom_point(size = 2.5) +
  geom_line(
    aes(group = Variant)
  ) +
  coord_equal() +
  labs(
    title = "Locked-Test Calibration",
    x = "Mean predicted probability",
    y = "Observed default rate"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "xgb_locked_test_calibration.png"),
  p_test_calibration,
  width = 7,
  height = 6,
  dpi = 300
)


# ==============================================================================
# 21. RESEARCH FAIRNESS / SUBGROUP DIAGNOSTICS
#
# IMPORTANT:
# These are model-governance diagnostics using demographic variables available
# in the benchmark dataset. They do NOT establish legal/regulatory compliance.
# ==============================================================================

subgroup_metrics <- function(
    group_vector,
    group_name,
    y,
    p,
    threshold
) {

  df <- tibble(
    Group = as.character(group_vector),
    y = y,
    p = p
  ) %>%
    filter(
      !is.na(Group),
      Group != ""
    )

  out <- df %>%
    group_by(Group) %>%
    group_modify(
      ~ {
        metrics <- threshold_metrics(
          .x$y,
          .x$p,
          threshold
        )

        tibble(
          N = nrow(.x),
          ObservedDefaultRate = mean(.x$y),
          MeanPredictedPD = mean(.x$p),
          ROC_AUC = roc_auc_value(.x$y, .x$p),
          Brier = brier_score(.x$y, .x$p),
          Recall = metrics$Recall,
          Specificity = metrics$Specificity,
          Precision = metrics$Precision,
          FPR = metrics$FPR,
          FNR = metrics$FNR,
          RiskFlagRate = metrics$RiskFlagRate
        )
      }
    ) %>%
    ungroup()

  # Research-only ratio against the largest subgroup's risk-flag rate.
  reference_group <- out %>%
    arrange(desc(N)) %>%
    slice(1) %>%
    pull(Group)

  reference_rate <- out %>%
    filter(Group == reference_group) %>%
    pull(RiskFlagRate)

  out %>%
    mutate(
      Attribute = group_name,
      ReferenceGroup = reference_group,
      RiskFlagRateRatio_vsReference = ifelse(
        is.na(reference_rate) || reference_rate == 0,
        NA_real_,
        RiskFlagRate / reference_rate
      )
    ) %>%
    relocate(
      Attribute,
      Group,
      ReferenceGroup
    )
}

age_band <- cut(
  test_data$Age,
  breaks = c(
    -Inf,
    24,
    34,
    44,
    54,
    64,
    Inf
  ),
  labels = c(
    "24 or younger",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+"
  ),
  ordered_result = TRUE
)

fairness_age <- subgroup_metrics(
  group_vector = age_band,
  group_name = "AgeBand",
  y = y_test,
  p = selected_test_prob,
  threshold = opt_threshold
)

fairness_marital <- subgroup_metrics(
  group_vector = test_data$MaritalStatus,
  group_name = "MaritalStatus",
  y = y_test,
  p = selected_test_prob,
  threshold = opt_threshold
)

fairness_diagnostics <- bind_rows(
  fairness_age,
  fairness_marital
)

write_csv(
  fairness_diagnostics,
  file.path(TABLE_DIR, "xgb_locked_test_subgroup_fairness_diagnostics.csv")
)


# ==============================================================================
# 22. FINAL SUMMARY + COMPARISON-READY MANIFEST
# ==============================================================================

selected_prob_row <- final_probability_metrics %>%
  filter(
    ProbabilityVariant == paste0(
      "Selected: ",
      selected_probability_variant
    )
  )

selected_threshold_row <- final_threshold_metrics %>%
  filter(
    ThresholdLabel == "Development-selected research threshold"
  )

final_summary <- tibble(
  Item = c(
    "Dataset MD5",
    "Full source observations",
    "Development sample",
    "Development rows",
    "Locked final-test rows",
    "XGBoost objective",
    "Primary CV metric",
    "CV folds",
    "Tuning candidates",
    "Best iteration",
    "Best CV PR-AUC",
    "Class weighting selected",
    "scale_pos_weight selected",
    "Probability variant selected",
    "Research threshold",
    "Locked-test ROC-AUC",
    "Locked-test PR-AUC",
    "Locked-test Brier",
    "Locked-test log loss",
    "Locked-test recall @ research threshold",
    "Locked-test specificity @ research threshold",
    "Locked-test precision @ research threshold",
    "Locked-test F1 @ research threshold",
    "Locked-test balanced accuracy @ research threshold",
    "OOT validation"
  ),
  Value = c(
    EXPECTED_MD5,
    "255347",
    "50000",
    as.character(nrow(dev_data)),
    as.character(nrow(test_data)),
    "binary:logistic",
    "PR-AUC / aucpr",
    as.character(N_FOLDS),
    as.character(nrow(candidate_grid)),
    as.character(best_iteration),
    sprintf("%.6f", best_candidate$CV_PR_AUC),
    as.character(best_candidate$scale_pos_weight != 1),
    sprintf("%.6f", best_candidate$scale_pos_weight),
    selected_probability_variant,
    sprintf("%.3f", opt_threshold),
    sprintf("%.6f", selected_prob_row$ROC_AUC),
    sprintf("%.6f", selected_prob_row$PR_AUC),
    sprintf("%.6f", selected_prob_row$Brier),
    sprintf("%.6f", selected_prob_row$LogLoss),
    sprintf("%.6f", selected_threshold_row$Recall),
    sprintf("%.6f", selected_threshold_row$Specificity),
    sprintf("%.6f", selected_threshold_row$Precision),
    sprintf("%.6f", selected_threshold_row$F1),
    sprintf("%.6f", selected_threshold_row$BalancedAccuracy),
    "Not performed: source dataset contains no genuine temporal variable."
  )
)

write_csv(
  final_summary,
  file.path(TABLE_DIR, "xgb_study2_final_summary.csv")
)

comparison_manifest <- tibble(
  Artifact = c(
    "Paired test row IDs",
    "Keyed XGBoost final-test probabilities",
    "Probability metrics",
    "Thresholded metrics",
    "Best XGBoost parameters"
  ),
  File = c(
    file.path(META_DIR, "locked_test_loanids_9999.csv"),
    file.path(TABLE_DIR, "xgb_locked_test_keyed_predictions.csv"),
    file.path(TABLE_DIR, "xgb_locked_test_probability_metrics.csv"),
    file.path(TABLE_DIR, "xgb_locked_test_classification_metrics.csv"),
    file.path(TABLE_DIR, "xgb_best_hyperparameters.csv")
  ),
  IntendedNextUse = c(
    "Verify exact pairing with frozen Logistic holdout rows",
    "Paired DeLong and paired bootstrap comparison",
    "Logistic-vs-XGBoost performance table",
    "Threshold-specific model comparison",
    "Reproducibility and Shiny deployment"
  )
)

write_csv(
  comparison_manifest,
  file.path(META_DIR, "model_comparison_ready_manifest.csv")
)


# ==============================================================================
# 23. COMPLETION MESSAGE
# ==============================================================================

cat("\n")
cat("======================================================================\n")
cat("STUDY 2 XGBOOST CHALLENGER COMPLETE\n")
cat("======================================================================\n")
cat(sprintf("Dataset MD5 verified              : %s\n", EXPECTED_MD5))
cat(sprintf("Development / locked test         : %d / %d\n",
            nrow(dev_data), nrow(test_data)))
cat(sprintf("Tuning candidates evaluated       : %d\n",
            nrow(candidate_grid)))
cat(sprintf("Best boosting iteration           : %d\n",
            best_iteration))
cat(sprintf("Best 5-fold CV PR-AUC             : %.4f\n",
            best_candidate$CV_PR_AUC))
cat(sprintf("Selected scale_pos_weight         : %.4f\n",
            best_candidate$scale_pos_weight))
cat(sprintf("Probability variant               : %s\n",
            selected_probability_variant))
cat(sprintf("Research threshold                : %.3f\n",
            opt_threshold))
cat(sprintf("Locked-test ROC-AUC               : %.4f\n",
            selected_prob_row$ROC_AUC))
cat(sprintf("Locked-test PR-AUC                : %.4f\n",
            selected_prob_row$PR_AUC))
cat(sprintf("Locked-test Recall                : %.4f\n",
            selected_threshold_row$Recall))
cat(sprintf("Locked-test Precision             : %.4f\n",
            selected_threshold_row$Precision))
cat(sprintf("Locked-test F1                    : %.4f\n",
            selected_threshold_row$F1))
cat(sprintf("Locked-test Balanced Accuracy     : %.4f\n",
            selected_threshold_row$BalancedAccuracy))
cat("SHAP outputs                      : generated\n")
cat("Subgroup fairness diagnostics     : generated\n")
cat("Logistic baseline                 : UNCHANGED\n")
cat("Logistic-vs-XGBoost comparison    : next separate study file\n")
cat("OOT validation                    : not claimed (no temporal field)\n")
cat(sprintf("Tables/figures output             : %s\n",
            normalizePath(OUTPUT_DIR)))
cat(sprintf("Deployment model artifacts        : %s\n",
            normalizePath(MODEL_DIR)))
cat("======================================================================\n")
