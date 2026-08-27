# =============================================================================
# Loan Default Risk Prediction Using Logistic Regression
# Thesis Reproducibility + Supporting Analysis + Shiny Application
#
# Author: Soumen Roy (Data Scientist, MSDS)
#
# Purpose
# -------
# This single R script reproduces the empirical analysis, tables, and analytical
# figures reported in the companion thesis monograph:
# "An End-to-End Explainable Loan Default Risk Prediction System in R:
#  Logistic Regression, Interactive Shiny Deployment, and an Industry-Grade
#  XGBoost Research Roadmap"
#
# Data source
# -----------
# NIKHIL. Loan Default Prediction Dataset. Kaggle.
# https://www.kaggle.com/datasets/nikhil1e9/loan-default
#
# IMPORTANT RESEARCH BOUNDARY
# ---------------------------
# - Logistic-regression results below are completed empirical results.
# - XGBoost, SHAP, calibration, fairness, and production-monitoring results in
#   the thesis are PROSPECTIVE research plans and are NOT fabricated here.
# - The 15% Shiny "High Risk" threshold is a demonstration heuristic, not a
#   validated production underwriting threshold.
# - Figure 9 (Shiny screenshot) and Figure 10 (Posit Connect configuration
#   screenshot) are interface/cloud-state screenshots rather than statistical
#   plots. This script reproduces the Shiny application itself; those two
#   screenshots are captured from the running application and Connect Cloud UI.
#
# Reproducibility
# ---------------
# Place this file in the same directory as Loan_default.csv and run it from top
# to bottom. All generated outputs are written to:
#   thesis_reproducibility_outputs/
#
# The script includes assertions against the numerical results reported in the
# thesis. If the source file, package behavior, factor levels, random sampling,
# or model outputs differ materially, execution stops with an explanatory error.
# =============================================================================


# =============================================================================
# 0. CONFIGURATION AND REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "tidyverse",
  "caret",
  "pROC",
  "shiny",
  "bslib",
  "bsicons"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required package(s): ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them first with:\n",
      "install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))"
    )
  )
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(pROC)
  library(shiny)
  library(bslib)
  library(bsicons)
})

DATA_FILE <- "Loan_default.csv"
OUTPUT_DIR <- "thesis_reproducibility_outputs"
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
MODEL_DIR <- file.path(OUTPUT_DIR, "models")
META_DIR <- file.path(OUTPUT_DIR, "metadata")

# Keep FALSE when reproducing thesis analysis non-interactively.
# Set TRUE only when you want to launch the Shiny application at the end.
RUN_SHINY_APP <- FALSE

dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(FIGURE_DIR, showWarnings = FALSE)
dir.create(TABLE_DIR, showWarnings = FALSE)
dir.create(MODEL_DIR, showWarnings = FALSE)
dir.create(META_DIR, showWarnings = FALSE)

if (!file.exists(DATA_FILE)) {
  stop(
    paste0(
      "Cannot find ", DATA_FILE, ".\n",
      "Place this R file in the same directory as Loan_default.csv ",
      "or change DATA_FILE to the correct path."
    )
  )
}

# Capture environment information for reproducibility.
capture.output(
  sessionInfo(),
  file = file.path(META_DIR, "sessionInfo.txt")
)

source_md5 <- unname(tools::md5sum(DATA_FILE))

writeLines(
  c(
    paste0("Data file: ", DATA_FILE),
    paste0("MD5: ", source_md5),
    "Expected MD5 for the thesis source snapshot: 5f5a3364753b86ef3472cba30af224ce",
    "Kaggle source: https://www.kaggle.com/datasets/nikhil1e9/loan-default"
  ),
  con = file.path(META_DIR, "data_source_and_fingerprint.txt")
)


# =============================================================================
# 1. LOAD THE FULL DATASET
#    Thesis Chapter 3: Data Provenance, Structure, and Quality
# =============================================================================

loans_raw <- read_csv(DATA_FILE, show_col_types = FALSE)

# The thesis was built from this exact benchmark snapshot.
EXPECTED_MD5 <- "5f5a3364753b86ef3472cba30af224ce"

if (!identical(source_md5, EXPECTED_MD5)) {
  stop(
    paste0(
      "Dataset fingerprint does not match the thesis snapshot.\n",
      "Expected MD5: ", EXPECTED_MD5, "\n",
      "Observed MD5: ", source_md5, "\n",
      "Use the same Loan_default.csv file if exact numerical reproduction is required."
    )
  )
}

# Structural checks reported in Thesis Table 3.
full_n <- nrow(loans_raw)
full_p <- ncol(loans_raw)
full_missing <- sum(is.na(loans_raw))
full_duplicates <- sum(duplicated(loans_raw))
full_defaults <- sum(loans_raw$Default == 1)
full_nondefaults <- sum(loans_raw$Default == 0)
full_default_prevalence <- mean(loans_raw$Default == 1)

structural_checks <- tibble(
  Check = c(
    "Observations",
    "Columns",
    "Missing cells",
    "Duplicate rows",
    "Observed defaults",
    "Observed non-defaults",
    "Default prevalence"
  ),
  Result = c(
    format(full_n, big.mark = ","),
    format(full_p, big.mark = ","),
    format(full_missing, big.mark = ","),
    format(full_duplicates, big.mark = ","),
    format(full_defaults, big.mark = ","),
    format(full_nondefaults, big.mark = ","),
    sprintf("%.2f%%", 100 * full_default_prevalence)
  )
)

write_csv(
  structural_checks,
  file.path(TABLE_DIR, "table_03_structural_quality_checks.csv")
)

# Exact full-dataset assertions from the thesis.
stopifnot(
  full_n == 255347,
  full_p == 18,
  full_missing == 0,
  full_duplicates == 0,
  full_defaults == 29653,
  full_nondefaults == 225694,
  round(100 * full_default_prevalence, 2) == 11.61
)


# =============================================================================
# 2. VARIABLE DICTIONARY
#    Thesis Table 4
# =============================================================================

variable_metadata <- tribble(
  ~Variable,          ~Type,                ~ResearchRole,
  "LoanID",           "Categorical / ID",   "Unique loan identifier; removed from modeling because it is an identifier rather than a risk predictor.",
  "Age",              "Numeric",            "Borrower age in years.",
  "Income",           "Numeric",            "Annual income in currency units represented by the dataset.",
  "LoanAmount",       "Numeric",            "Requested or recorded loan amount.",
  "CreditScore",      "Numeric",            "Numeric credit score.",
  "MonthsEmployed",   "Numeric",            "Length of employment in months.",
  "NumCreditLines",   "Numeric",            "Number of credit lines.",
  "InterestRate",     "Numeric",            "Loan interest rate, represented in percentage-point units.",
  "LoanTerm",         "Numeric",            "Loan term in months.",
  "DTIRatio",         "Numeric",            "Debt-to-income ratio.",
  "Education",        "Categorical",        "Borrower education category.",
  "EmploymentType",   "Categorical",        "Borrower employment category.",
  "MaritalStatus",    "Categorical",        "Borrower marital-status category.",
  "HasMortgage",      "Categorical",        "Whether the borrower has a mortgage.",
  "HasDependents",    "Categorical",        "Whether the borrower has dependents.",
  "LoanPurpose",      "Categorical",        "Recorded purpose of the loan.",
  "HasCoSigner",      "Categorical",        "Whether the loan has a co-signer.",
  "Default",          "Binary target",      "Observed target: 1 = default, 0 = non-default."
)

variable_dictionary <- variable_metadata %>%
  mutate(
    Unique = map_int(
      Variable,
      ~ n_distinct(loans_raw[[.x]])
    )
  ) %>%
  select(Variable, Type, Unique, ResearchRole)

write_csv(
  variable_dictionary,
  file.path(TABLE_DIR, "table_04_variable_dictionary.csv")
)


# =============================================================================
# 3. FULL-DATASET DESCRIPTIVE STATISTICS
#    Thesis Table 5
# =============================================================================

numeric_vars <- c(
  "Age",
  "Income",
  "LoanAmount",
  "CreditScore",
  "MonthsEmployed",
  "NumCreditLines",
  "InterestRate",
  "LoanTerm",
  "DTIRatio"
)

numeric_summary <- map_dfr(
  numeric_vars,
  function(v) {
    x <- loans_raw[[v]]
    tibble(
      Variable = v,
      Mean = mean(x),
      SD = sd(x),
      Min = min(x),
      Q1 = as.numeric(quantile(x, 0.25, type = 7)),
      Median = median(x),
      Q3 = as.numeric(quantile(x, 0.75, type = 7)),
      Max = max(x)
    )
  }
)

numeric_summary_display <- numeric_summary %>%
  mutate(
    across(
      c(Mean, SD, Min, Q1, Median, Q3, Max),
      ~ round(.x, 3)
    )
  )

write_csv(
  numeric_summary_display,
  file.path(TABLE_DIR, "table_05_numeric_descriptive_statistics.csv")
)


# =============================================================================
# 4. FULL-DATASET NUMERIC SEPARATION BY OUTCOME
#    Thesis Table 6 and Figure 3
# =============================================================================

pooled_sd <- function(x0, x1) {
  sqrt(
    (
      (length(x0) - 1) * var(x0) +
        (length(x1) - 1) * var(x1)
    ) /
      (length(x0) + length(x1) - 2)
  )
}

numeric_group_differences <- map_dfr(
  numeric_vars,
  function(v) {
    x0 <- loans_raw %>%
      filter(Default == 0) %>%
      pull(all_of(v))

    x1 <- loans_raw %>%
      filter(Default == 1) %>%
      pull(all_of(v))

    mean0 <- mean(x0)
    mean1 <- mean(x1)
    difference <- mean1 - mean0
    d <- difference / pooled_sd(x0, x1)

    tibble(
      Variable = v,
      Mean_NonDefault = mean0,
      Mean_Default = mean1,
      Difference_DefaultMinusNonDefault = difference,
      CohensD = d
    )
  }
)

numeric_group_differences_display <- numeric_group_differences %>%
  mutate(
    across(
      c(
        Mean_NonDefault,
        Mean_Default,
        Difference_DefaultMinusNonDefault,
        CohensD
      ),
      ~ round(.x, 3)
    )
  )

write_csv(
  numeric_group_differences_display,
  file.path(TABLE_DIR, "table_06_numeric_group_differences.csv")
)

figure_03_data <- numeric_group_differences %>%
  mutate(
    Variable = factor(
      Variable,
      levels = rev(numeric_vars)
    )
  )

p03 <- ggplot(
  figure_03_data,
  aes(x = Variable, y = CohensD)
) +
  geom_col(width = 0.72) +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  geom_text(
    aes(
      label = sprintf("%.2f", CohensD),
      hjust = if_else(CohensD >= 0, -0.15, 1.15)
    ),
    size = 3.7
  ) +
  coord_flip() +
  scale_y_continuous(
    expand = expansion(mult = c(0.17, 0.17))
  ) +
  labs(
    title = "Univariate Separation Between Default and Non-default Groups",
    x = NULL,
    y = "Standardized mean difference (Default - Non-default), Cohen's d"
  ) +
  theme_classic(base_size = 12)

ggsave(
  file.path(FIGURE_DIR, "figure_03_standardized_mean_differences.png"),
  p03,
  width = 10,
  height = 6,
  dpi = 300
)


# =============================================================================
# 5. FULL-DATASET CATEGORICAL DEFAULT RATES
#    Thesis Tables 7-13 and Figures 4-5
# =============================================================================

categorical_vars <- c(
  "Education",
  "EmploymentType",
  "MaritalStatus",
  "HasMortgage",
  "HasDependents",
  "LoanPurpose",
  "HasCoSigner"
)

categorical_default_rate <- function(data, variable) {
  out <- data %>%
    group_by(Category = .data[[variable]]) %>%
    summarise(
      N = n(),
      ObservedDefaultRate = mean(Default == 1),
      .groups = "drop"
    ) %>%
    arrange(desc(ObservedDefaultRate))

  names(out)[1] <- variable
  out
}

categorical_rate_tables <- setNames(
  lapply(
    categorical_vars,
    function(v) categorical_default_rate(loans_raw, v)
  ),
  categorical_vars
)

table_number_map <- c(
  Education = "07",
  EmploymentType = "08",
  MaritalStatus = "09",
  HasMortgage = "10",
  HasDependents = "11",
  LoanPurpose = "12",
  HasCoSigner = "13"
)

for (v in categorical_vars) {
  display_table <- categorical_rate_tables[[v]]
  display_table$ObservedDefaultRatePercent <- round(
    100 * display_table$ObservedDefaultRate,
    2
  )
  display_table$ObservedDefaultRate <- NULL

  write_csv(
    display_table,
    file.path(
      TABLE_DIR,
      paste0(
        "table_",
        table_number_map[[v]],
        "_",
        tolower(v),
        "_default_rates.csv"
      )
    )
  )
}

# Figure 4: Employment type default rates
employment_rates <- categorical_rate_tables[["EmploymentType"]] %>%
  mutate(
    EmploymentType = factor(
      EmploymentType,
      levels = c(
        "Unemployed",
        "Part-time",
        "Self-employed",
        "Full-time"
      )
    ),
    DefaultRatePercent = 100 * ObservedDefaultRate
  )

p04 <- ggplot(
  employment_rates,
  aes(x = EmploymentType, y = DefaultRatePercent)
) +
  geom_col(width = 0.75) +
  geom_text(
    aes(label = sprintf("%.1f%%", DefaultRatePercent)),
    vjust = -0.35,
    size = 3.8
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))
  ) +
  labs(
    title = "Observed Default Rate by Employment Type",
    x = NULL,
    y = "Observed default rate (%)"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 18, hjust = 1)
  )

ggsave(
  file.path(FIGURE_DIR, "figure_04_default_rate_by_employment_type.png"),
  p04,
  width = 8,
  height = 5.3,
  dpi = 300
)

# Figure 5: Education default rates
education_rates <- categorical_rate_tables[["Education"]] %>%
  mutate(
    Education = factor(
      Education,
      levels = c(
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
      )
    ),
    DefaultRatePercent = 100 * ObservedDefaultRate
  )

p05 <- ggplot(
  education_rates,
  aes(x = Education, y = DefaultRatePercent)
) +
  geom_col(width = 0.75) +
  geom_text(
    aes(label = sprintf("%.1f%%", DefaultRatePercent)),
    vjust = -0.35,
    size = 3.8
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))
  ) +
  labs(
    title = "Observed Default Rate by Education Level",
    x = NULL,
    y = "Observed default rate (%)"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 18, hjust = 1)
  )

ggsave(
  file.path(FIGURE_DIR, "figure_05_default_rate_by_education_level.png"),
  p05,
  width = 8,
  height = 5.3,
  dpi = 300
)


# =============================================================================
# 6. FIGURE 1: FULL-DATASET CLASS DISTRIBUTION
# =============================================================================

class_distribution <- loans_raw %>%
  count(Default, name = "N") %>%
  arrange(Default) %>%
  mutate(
    Percent = N / sum(N),
    ClassLabel = if_else(
      Default == 0,
      "Non-default (0)",
      "Default (1)"
    ),
    BarLabel = paste0(
      format(N, big.mark = ","),
      "\n(",
      sprintf("%.1f%%", 100 * Percent),
      ")"
    )
  )

p01 <- ggplot(
  class_distribution,
  aes(x = ClassLabel, y = N)
) +
  geom_col(width = 0.80) +
  geom_text(
    aes(label = BarLabel),
    vjust = -0.35,
    size = 4.2
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.16))
  ) +
  labs(
    title = "Class Distribution in the Full Loan Dataset",
    x = NULL,
    y = "Number of observations"
  ) +
  theme_classic(base_size = 14)

ggsave(
  file.path(FIGURE_DIR, "figure_01_full_dataset_class_distribution.png"),
  p01,
  width = 10,
  height = 6.3,
  dpi = 300
)


# =============================================================================
# 7. FIGURE 2: CURRENT END-TO-END ANALYTICAL + DEPLOYMENT WORKFLOW
# =============================================================================

draw_flowchart <- function(nodes, edges, title_text, output_file,
                           width = 12, height = 6) {

  edge_plot <- edges %>%
    left_join(
      nodes %>%
        select(from = id, x_from = x, y_from = y),
      by = "from"
    ) %>%
    left_join(
      nodes %>%
        select(to = id, x_to = x, y_to = y),
      by = "to"
    )

  p <- ggplot() +
    geom_segment(
      data = edge_plot,
      aes(
        x = x_from,
        y = y_from,
        xend = x_to,
        yend = y_to
      ),
      arrow = grid::arrow(
        length = grid::unit(0.14, "inches"),
        type = "closed"
      ),
      linewidth = 0.55
    ) +
    geom_rect(
      data = nodes,
      aes(
        xmin = x - width / 2,
        xmax = x + width / 2,
        ymin = y - height / 2,
        ymax = y + height / 2
      ),
      fill = "white",
      linewidth = 0.7
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      size = 3.3,
      lineheight = 0.95
    ) +
    coord_cartesian(
      xlim = c(min(nodes$x) - 1.1, max(nodes$x) + 1.1),
      ylim = c(min(nodes$y) - 1.0, max(nodes$y) + 1.0),
      clip = "off"
    ) +
    labs(title = title_text) +
    theme_void(base_size = 12) +
    theme(
      plot.title = element_text(
        hjust = 0.5,
        face = "plain",
        margin = margin(b = 15)
      )
    )

  ggsave(
    output_file,
    p,
    width = width,
    height = height,
    dpi = 300
  )

  invisible(p)
}

workflow_nodes <- tribble(
  ~id, ~label,                                      ~x,  ~y, ~width, ~height,
  "a", "Kaggle CSV\n255,347 x 18",                 1.0, 3.2, 1.55, 0.72,
  "b", "Deterministic sample\n50,000 rows",         3.1, 3.2, 1.75, 0.72,
  "c", "Clean + factor\nencoding",                  5.3, 3.2, 1.55, 0.72,
  "d", "Stratified 80/20\nsplit",                   7.5, 3.2, 1.55, 0.72,
  "e", "Logistic regression\n+ stepwise AIC",       3.0, 1.8, 1.85, 0.72,
  "f", "Evaluation\nROC / confusion matrix",        5.5, 1.8, 1.90, 0.72,
  "g", "Interactive\nrisk assessment",              2.7, 0.45, 1.70, 0.72,
  "h", "Shiny + GitHub +\nPosit Connect Cloud",     6.9, 0.45, 2.05, 0.72
)

workflow_edges <- tribble(
  ~from, ~to,
  "a", "b",
  "b", "c",
  "c", "d",
  "d", "e",
  "e", "f",
  "f", "h",
  "h", "g"
)

p02 <- draw_flowchart(
  workflow_nodes,
  workflow_edges,
  "Current End-to-End Analytical and Deployment Workflow",
  file.path(FIGURE_DIR, "figure_02_current_end_to_end_workflow.png"),
  width = 12,
  height = 5.2
)


# =============================================================================
# 8. CREATE THE EXACT 50,000-ROW DEVELOPMENT SAMPLE
#    This is the same core sampling logic used in the user's app.R.
# =============================================================================

set.seed(1)
loan_data <- dplyr::sample_n(loans_raw, size = 50000)

# Full-dataset empirical outputs have now been created, so release the larger
# object just as the deployed workflow does after sampling.
rm(loans_raw)
gc(verbose = FALSE)

stopifnot(nrow(loan_data) == 50000)


# =============================================================================
# 9. CLEAN DATA + STRATIFIED TRAIN/TEST SPLIT
#    This section directly aligns with the supplied app.R.
# =============================================================================

data_clean <- loan_data %>%
  select(-LoanID) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(Default = as.factor(Default))

rm(loan_data)
gc(verbose = FALSE)

# Confirm target factor levels are exactly 0 and 1.
stopifnot(identical(levels(data_clean$Default), c("0", "1")))

set.seed(123)
trainIndex <- createDataPartition(
  data_clean$Default,
  p = 0.8,
  list = FALSE
)

train_data <- data_clean[trainIndex, ]
test_data <- data_clean[-trainIndex, ]

# The thesis reports 40,001 training observations and 9,999 holdout observations.
stopifnot(
  nrow(train_data) == 40001,
  nrow(test_data) == 9999
)


# =============================================================================
# 10. FIT FULL LOGISTIC REGRESSION + BIDIRECTIONAL STEPWISE AIC
#     Thesis Chapter 6
# =============================================================================

full_model <- glm(
  Default ~ .,
  data = train_data,
  family = "binomial"
)

model_stepwise <- step(
  full_model,
  direction = "both",
  trace = 0
)

# Save model artifacts so the fitted objects themselves are reproducible outputs.
saveRDS(
  full_model,
  file.path(MODEL_DIR, "full_logistic_model.rds")
)

saveRDS(
  model_stepwise,
  file.path(MODEL_DIR, "stepwise_logistic_model.rds")
)

# Thesis Table 14: Full vs. stepwise model fit.
model_fit_comparison <- tibble(
  Model = c(
    "Full logistic model",
    "Stepwise logistic model"
  ),
  NullDeviance = c(
    full_model$null.deviance,
    model_stepwise$null.deviance
  ),
  ResidualDeviance = c(
    deviance(full_model),
    deviance(model_stepwise)
  ),
  AIC = c(
    AIC(full_model),
    AIC(model_stepwise)
  ),
  Specification = c(
    "Includes LoanTerm",
    "LoanTerm removed"
  )
)

model_fit_display <- model_fit_comparison %>%
  mutate(
    NullDeviance = round(NullDeviance),
    ResidualDeviance = round(ResidualDeviance),
    AIC = round(AIC)
  )

write_csv(
  model_fit_display,
  file.path(TABLE_DIR, "table_14_full_vs_stepwise_model_fit.csv")
)

# Expected thesis fit statistics.
stopifnot(
  round(AIC(full_model)) == 25360,
  round(AIC(model_stepwise)) == 25358,
  round(deviance(full_model)) == 25310,
  round(deviance(model_stepwise)) == 25310
)

selected_predictors <- attr(
  terms(model_stepwise),
  "term.labels"
)

EXPECTED_SELECTED_PREDICTORS <- c(
  "Age",
  "Income",
  "LoanAmount",
  "CreditScore",
  "MonthsEmployed",
  "NumCreditLines",
  "InterestRate",
  "DTIRatio",
  "Education",
  "EmploymentType",
  "MaritalStatus",
  "HasMortgage",
  "HasDependents",
  "LoanPurpose",
  "HasCoSigner"
)

stopifnot(
  setequal(selected_predictors, EXPECTED_SELECTED_PREDICTORS),
  !("LoanTerm" %in% selected_predictors),
  length(selected_predictors) == 15
)


# =============================================================================
# 11. STEPWISE COEFFICIENT TABLE
#     Thesis Table 15
# =============================================================================

coef_matrix <- summary(model_stepwise)$coefficients

stepwise_coefficient_table <- as.data.frame(coef_matrix) %>%
  rownames_to_column("Term") %>%
  as_tibble() %>%
  rename(
    Estimate = Estimate,
    SE = `Std. Error`,
    Z = `z value`,
    PValue = `Pr(>|z|)`
  )

write_csv(
  stepwise_coefficient_table,
  file.path(TABLE_DIR, "table_15_stepwise_model_coefficients.csv")
)


# =============================================================================
# 12. SCALED ODDS RATIOS + 95% WALD CONFIDENCE INTERVALS
#     Thesis Table 16 and Figure 6
# =============================================================================

scaled_effect_specs <- tribble(
  ~Predictor,         ~IncrementNumeric, ~Increment,              ~FigureLabel,
  "Age",                         10,      "+10 years",             "Age (10 years)",
  "Income",                   10000,      "+$10,000",              "Income ($10k)",
  "LoanAmount",               10000,      "+$10,000",              "Loan amount ($10k)",
  "CreditScore",                 50,      "+50 points",            "Credit score (50 pts)",
  "MonthsEmployed",              12,      "+12 months",            "Employment (12 months)",
  "NumCreditLines",               1,      "+1 line",               "Credit lines (+1)",
  "InterestRate",                 1,      "+1 percentage point",   "Interest rate (+1 pp)",
  "DTIRatio",                   0.10,      "+0.10",                 "DTI (+0.10)"
)

scaled_odds_ratios <- scaled_effect_specs %>%
  rowwise() %>%
  mutate(
    Beta = coef_matrix[Predictor, "Estimate"],
    SE = coef_matrix[Predictor, "Std. Error"],
    OddsRatio = exp(Beta * IncrementNumeric),
    CI_Low = exp(
      (Beta - 1.96 * SE) * IncrementNumeric
    ),
    CI_High = exp(
      (Beta + 1.96 * SE) * IncrementNumeric
    ),
    Direction = if_else(
      OddsRatio < 1,
      "Lower conditional odds",
      "Higher conditional odds"
    )
  ) %>%
  ungroup()

scaled_odds_ratios_display <- scaled_odds_ratios %>%
  transmute(
    Predictor,
    Increment,
    OddsRatio = round(OddsRatio, 3),
    `95% CI low` = round(CI_Low, 3),
    `95% CI high` = round(CI_High, 3),
    Direction
  )

write_csv(
  scaled_odds_ratios_display,
  file.path(TABLE_DIR, "table_16_scaled_odds_ratios.csv")
)

# Figure 6
figure_06_data <- scaled_odds_ratios %>%
  mutate(
    FigureLabel = factor(
      FigureLabel,
      levels = rev(scaled_effect_specs$FigureLabel)
    )
  )

p06 <- ggplot(
  figure_06_data,
  aes(x = FigureLabel, y = OddsRatio)
) +
  geom_hline(
    yintercept = 1,
    linewidth = 0.5
  ) +
  geom_errorbar(
    aes(ymin = CI_Low, ymax = CI_High),
    width = 0.16,
    linewidth = 0.7
  ) +
  geom_point(size = 2.7) +
  coord_flip() +
  labs(
    title = "Stepwise Logistic Model: Selected Numeric Effects",
    x = NULL,
    y = "Odds ratio (scaled increment; 95% Wald CI)"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "figure_06_scaled_odds_ratios.png"),
  p06,
  width = 9.5,
  height = 5.8,
  dpi = 300
)


# =============================================================================
# 13. HOLDOUT PREDICTIONS FROM THE FULL LOGISTIC BASELINE
#     Thesis Chapter 7
#
# IMPORTANT:
# The thesis holdout performance numbers are based on the FULL logistic model.
# The stepwise model is used for the deployed Shiny demonstration.
# =============================================================================

probabilities <- predict(
  full_model,
  newdata = test_data,
  type = "response"
)

predictions <- ifelse(
  probabilities > 0.5,
  "1",
  "0"
)

predictions <- factor(
  predictions,
  levels = levels(test_data$Default)
)

holdout_predictions <- tibble(
  ActualDefault = test_data$Default,
  PredictedProbability = as.numeric(probabilities),
  PredictedClass_050 = predictions
)

write_csv(
  holdout_predictions,
  file.path(TABLE_DIR, "holdout_row_level_predictions_threshold_050.csv")
)


# =============================================================================
# 14. CONFUSION MATRIX + CLASSIFICATION METRICS
#     Thesis Tables 17-18 and Figure 7
# =============================================================================

conf_matrix <- confusionMatrix(
  predictions,
  test_data$Default,
  positive = "1"
)

# Exact confusion counts.
TN <- unname(conf_matrix$table["0", "0"])
FN <- unname(conf_matrix$table["0", "1"])
FP <- unname(conf_matrix$table["1", "0"])
TP <- unname(conf_matrix$table["1", "1"])

stopifnot(
  TN == 8781,
  FN == 1133,
  FP == 39,
  TP == 46
)

confusion_matrix_table <- tibble(
  Prediction = c(
    "Predicted non-default (0)",
    "Predicted default (1)"
  ),
  `Actual non-default (0)` = c(TN, FP),
  `Actual default (1)` = c(FN, TP)
)

write_csv(
  confusion_matrix_table,
  file.path(TABLE_DIR, "table_17_holdout_confusion_matrix_threshold_050.csv")
)

# ROC and AUC from continuous probabilities.
roc_curve <- pROC::roc(
  response = test_data$Default,
  predictor = probabilities,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

roc_auc <- as.numeric(pROC::auc(roc_curve))

accuracy <- unname(conf_matrix$overall["Accuracy"])
accuracy_lower <- unname(conf_matrix$overall["AccuracyLower"])
accuracy_upper <- unname(conf_matrix$overall["AccuracyUpper"])
nir <- unname(conf_matrix$overall["AccuracyNull"])
accuracy_p <- unname(conf_matrix$overall["AccuracyPValue"])
kappa <- unname(conf_matrix$overall["Kappa"])

default_recall <- unname(conf_matrix$byClass["Sensitivity"])
specificity <- unname(conf_matrix$byClass["Specificity"])
default_precision <- unname(conf_matrix$byClass["Pos Pred Value"])
balanced_accuracy <- unname(conf_matrix$byClass["Balanced Accuracy"])

# Numerical assertions against the thesis.
stopifnot(
  round(accuracy, 4) == 0.8828,
  round(nir, 4) == 0.8821,
  round(accuracy_p, 4) == 0.4216,
  round(kappa, 4) == 0.0578,
  round(default_recall, 4) == 0.0390,
  round(specificity, 4) == 0.9956,
  round(default_precision, 4) == 0.5412,
  round(balanced_accuracy, 4) == 0.5173,
  round(roc_auc, 3) == 0.751
)

metrics_table <- tibble(
  Metric = c(
    "Accuracy",
    "95% CI for accuracy",
    "No Information Rate",
    "p-value: Accuracy > NIR",
    "Kappa",
    "Default sensitivity / recall",
    "Specificity",
    "Precision for default",
    "Balanced accuracy",
    "ROC-AUC"
  ),
  Value = c(
    sprintf("%.4f", accuracy),
    paste0(
      sprintf("%.4f", accuracy_lower),
      "-",
      sprintf("%.4f", accuracy_upper)
    ),
    sprintf("%.4f", nir),
    sprintf("%.4f", accuracy_p),
    sprintf("%.4f", kappa),
    sprintf("%.4f", default_recall),
    sprintf("%.4f", specificity),
    sprintf("%.4f", default_precision),
    sprintf("%.4f", balanced_accuracy),
    sprintf("%.3f", roc_auc)
  ),
  Interpretation = c(
    "High in isolation, but essentially the majority-class rate.",
    "Sampling interval for thresholded accuracy.",
    "Accuracy of always predicting the majority class.",
    "No evidence that threshold-0.50 accuracy exceeds the majority baseline.",
    "Very weak agreement beyond chance.",
    "Only 3.90% of actual defaults detected.",
    "99.56% of non-defaults correctly classified.",
    "Among predicted defaults, about 54.1% are actual defaults.",
    "Only slightly above random 0.50 when classes are weighted equally.",
    "Moderate ranking discrimination across thresholds."
  )
)

write_csv(
  metrics_table,
  file.path(TABLE_DIR, "table_18_holdout_performance_metrics.csv")
)

# Figure 7: confusion matrix heatmap
figure_07_data <- tribble(
  ~Predicted,      ~Actual,      ~Count,
  "Predicted 0",   "Actual 0",   TN,
  "Predicted 0",   "Actual 1",   FN,
  "Predicted 1",   "Actual 0",   FP,
  "Predicted 1",   "Actual 1",   TP
) %>%
  mutate(
    Predicted = factor(
      Predicted,
      levels = c("Predicted 1", "Predicted 0")
    ),
    Actual = factor(
      Actual,
      levels = c("Actual 0", "Actual 1")
    )
  )

p07 <- ggplot(
  figure_07_data,
  aes(x = Actual, y = Predicted, fill = Count)
) +
  geom_tile() +
  geom_text(
    aes(label = format(Count, big.mark = ",")),
    size = 5
  ) +
  labs(
    title = "Confusion Matrix at Threshold 0.50\n50,000-row Development Sample",
    x = NULL,
    y = NULL,
    fill = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  file.path(FIGURE_DIR, "figure_07_holdout_confusion_matrix_threshold_050.png"),
  p07,
  width = 7,
  height = 6,
  dpi = 300
)


# =============================================================================
# 15. FIGURE 8: WHY ACCURACY ALONE IS MISLEADING
# =============================================================================

figure_08_metrics <- tibble(
  Metric = c(
    "Accuracy",
    "Default recall",
    "Specificity",
    "Balanced accuracy",
    "ROC-AUC"
  ),
  Value = c(
    accuracy,
    default_recall,
    specificity,
    balanced_accuracy,
    roc_auc
  )
) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c(
        "Accuracy",
        "Default recall",
        "Specificity",
        "Balanced accuracy",
        "ROC-AUC"
      )
    )
  )

p08 <- ggplot(
  figure_08_metrics,
  aes(x = Metric, y = Value)
) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    vjust = -0.35,
    size = 3.8
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Why Accuracy Alone Is Misleading Under Class Imbalance",
    x = NULL,
    y = "Metric value"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 22, hjust = 1)
  )

ggsave(
  file.path(FIGURE_DIR, "figure_08_selected_holdout_metrics.png"),
  p08,
  width = 9,
  height = 5.5,
  dpi = 300
)


# =============================================================================
# 16. THESIS TABLE 26: COMPLETE REPRODUCIBILITY SNAPSHOT
# =============================================================================

reproducibility_snapshot <- tibble(
  Item = c(
    "Source dataset",
    "Full dimensions",
    "Full default prevalence",
    "Development sample",
    "Partition",
    "Baseline algorithm",
    "Selection",
    "Selected variables",
    "Holdout N",
    "ROC-AUC",
    "Accuracy at 0.50",
    "Default recall at 0.50",
    "Specificity at 0.50",
    "Balanced accuracy at 0.50",
    "Public app"
  ),
  Value = c(
    "Loan_default.csv; Kaggle Loan Default Prediction Dataset",
    "255,347 rows x 18 columns",
    "11.61%",
    "50,000 rows; set.seed(1)",
    "Stratified 80/20; set.seed(123)",
    'R glm(..., family = "binomial")',
    'step(..., direction = "both")',
    "15 predictors; LoanTerm removed",
    format(nrow(test_data), big.mark = ","),
    sprintf("%.3f", roc_auc),
    sprintf("%.4f", accuracy),
    sprintf("%.4f", default_recall),
    sprintf("%.4f", specificity),
    sprintf("%.4f", balanced_accuracy),
    "https://01a03965-4776-61cb-5830-8f5513dfa419.share.connect.posit.cloud/"
  )
)

write_csv(
  reproducibility_snapshot,
  file.path(TABLE_DIR, "table_26_complete_reproducibility_snapshot.csv")
)


# =============================================================================
# 17. FIGURE 11: PROPOSED INDUSTRY-GRADE XGBOOST RESEARCH + MLOPS ROADMAP
#
# This figure is a RESEARCH PLAN only. No XGBoost performance is generated or
# claimed by this script because the thesis explicitly labels it prospective.
# =============================================================================

xgb_nodes <- tribble(
  ~id, ~label,                                      ~x,  ~y, ~width, ~height,
  "a", "Full dataset\n+ frozen test set",           1.0, 3.2, 1.85, 0.72,
  "b", "Train / validation / test\nor nested CV",   3.4, 3.2, 2.15, 0.72,
  "c", "Encoding + imbalance\nstrategy",            5.9, 3.2, 1.95, 0.72,
  "d", "XGBoost tuning +\nearly stopping",          1.8, 1.75, 1.85, 0.72,
  "e", "Threshold + calibration\n(PR-AUC, Brier)",  4.1, 1.75, 2.05, 0.72,
  "f", "SHAP + stability +\nsubgroup diagnostics",  6.5, 1.75, 2.15, 0.72,
  "g", "Locked final test\ncomparison",             3.0, 0.35, 1.85, 0.72,
  "h", "Model registry + Shiny\n+ monitoring",      5.7, 0.35, 2.05, 0.72
)

xgb_edges <- tribble(
  ~from, ~to,
  "a", "b",
  "b", "c",
  "c", "d",
  "d", "e",
  "e", "f",
  "f", "g",
  "g", "h"
)

p11 <- draw_flowchart(
  xgb_nodes,
  xgb_edges,
  "Proposed Industry-Grade XGBoost Research and MLOps Roadmap",
  file.path(FIGURE_DIR, "figure_11_xgboost_research_mlops_roadmap.png"),
  width = 12,
  height = 5.5
)


# =============================================================================
# 18. DOCUMENT-TO-CODE ALIGNMENT MANIFEST
#
# Figures 1-8 and 11 are generated directly from this script.
# Figure 9 and Figure 10 are screenshots of external UI states rather than
# statistical plots. Their underlying application/configuration is documented.
# =============================================================================

alignment_manifest <- tibble(
  ThesisArtifact = c(
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 4",
    "Figure 5",
    "Figure 6",
    "Figure 7",
    "Figure 8",
    "Figure 9",
    "Figure 10",
    "Figure 11",
    "Tables 3-13",
    "Tables 14-18",
    "Table 26"
  ),
  CodeSection = c(
    "Section 6",
    "Section 7",
    "Section 4",
    "Section 5",
    "Section 5",
    "Section 12",
    "Section 14",
    "Section 15",
    "Section 20 (launch current Shiny app)",
    "External Posit Connect Cloud configuration screenshot",
    "Section 17",
    "Sections 1-5",
    "Sections 10-14",
    "Section 16"
  ),
  ReproductionStatus = c(
    rep("Generated directly by R", 8),
    "Application itself reproduced; screenshot captured from running UI",
    "External cloud-state screenshot; values are documented, not model-generated",
    "Generated directly by R",
    "Generated directly by R",
    "Generated directly by R",
    "Generated directly by R"
  )
)

write_csv(
  alignment_manifest,
  file.path(META_DIR, "thesis_document_to_code_alignment_manifest.csv")
)

writeLines(
  c(
    "FIGURE 9 - SHINY SCREENSHOT",
    "Set RUN_SHINY_APP <- TRUE and run this script.",
    "The script launches the current Loan Default Risk Analyzer.",
    "Capture the running browser interface if a static screenshot is required.",
    "",
    "The thesis labels its Figure 9 as a development-stage interface screenshot",
    "and explicitly states that it is not necessarily pixel-identical to the",
    "current deployed version."
  ),
  file.path(META_DIR, "figure_09_shiny_capture_instructions.txt")
)

writeLines(
  c(
    "FIGURE 10 - POSIT CONNECT CLOUD SCREENSHOT",
    "This is an external deployment-configuration screenshot rather than a",
    "statistical figure generated by R.",
    "",
    "Documented settings:",
    "Repository: sroy-codebase/projects",
    "Branch: main",
    "Primary file: loan-default-risk-prediction/app.R",
    "Public app: https://01a03965-4776-61cb-5830-8f5513dfa419.share.connect.posit.cloud/"
  ),
  file.path(META_DIR, "figure_10_posit_connect_capture_instructions.txt")
)


# =============================================================================
# 19. SAVE A MACHINE-READABLE VERIFICATION SUMMARY
# =============================================================================

verification_summary <- tibble(
  Check = c(
    "Dataset MD5",
    "Full observations",
    "Full columns",
    "Defaults",
    "Non-defaults",
    "Full default prevalence",
    "Development sample N",
    "Training N",
    "Holdout N",
    "Full-model rounded AIC",
    "Stepwise-model rounded AIC",
    "Selected predictors",
    "LoanTerm removed",
    "TN",
    "FN",
    "FP",
    "TP",
    "Accuracy",
    "Default recall",
    "Specificity",
    "Balanced accuracy",
    "ROC-AUC"
  ),
  Observed = c(
    source_md5,
    as.character(full_n),
    as.character(full_p),
    as.character(full_defaults),
    as.character(full_nondefaults),
    sprintf("%.4f", full_default_prevalence),
    "50000",
    as.character(nrow(train_data)),
    as.character(nrow(test_data)),
    as.character(round(AIC(full_model))),
    as.character(round(AIC(model_stepwise))),
    as.character(length(selected_predictors)),
    as.character(!("LoanTerm" %in% selected_predictors)),
    as.character(TN),
    as.character(FN),
    as.character(FP),
    as.character(TP),
    sprintf("%.4f", accuracy),
    sprintf("%.4f", default_recall),
    sprintf("%.4f", specificity),
    sprintf("%.4f", balanced_accuracy),
    sprintf("%.6f", roc_auc)
  ),
  ThesisExpected = c(
    EXPECTED_MD5,
    "255347",
    "18",
    "29653",
    "225694",
    "0.1161",
    "50000",
    "40001",
    "9999",
    "25360",
    "25358",
    "15",
    "TRUE",
    "8781",
    "1133",
    "39",
    "46",
    "0.8828",
    "0.0390",
    "0.9956",
    "0.5173",
    "0.751284"
  )
)

write_csv(
  verification_summary,
  file.path(META_DIR, "verification_summary.csv")
)


# =============================================================================
# 20. CURRENT SHINY APPLICATION
#     This preserves and aligns with the user's supplied app.R logic.
#
# Set RUN_SHINY_APP <- TRUE in Section 0 to launch it.
# =============================================================================

if (RUN_SHINY_APP) {

  ui <- page_sidebar(
    theme = bs_theme(bootswatch = "flatly"),
    title = "Loan Default Risk Analyzer",

    sidebar = sidebar(
      title = "Borrower Inputs",

      numericInput(
        "age",
        "Age",
        value = 35,
        min = 18
      ),

      numericInput(
        "income",
        "Annual Income ($)",
        value = 60000
      ),

      numericInput(
        "loanAmt",
        "Loan Amount ($)",
        value = 20000
      ),

      numericInput(
        "creditScore",
        "Credit Score",
        value = 700,
        min = 300,
        max = 850
      ),

      numericInput(
        "monthsEmp",
        "Months Employed",
        value = 48
      ),

      numericInput(
        "interest",
        "Interest Rate (%)",
        value = 6.5
      ),

      selectInput(
        "education",
        "Education Level",
        choices = levels(data_clean$Education)
      ),

      selectInput(
        "employment",
        "Employment Type",
        choices = levels(data_clean$EmploymentType)
      ),

      hr(),

      actionButton(
        "predict",
        "Generate Assessment",
        class = "btn-primary w-100"
      )
    ),

    layout_column_wrap(
      width = 1 / 2,

      value_box(
        title = "Calculated Probability",
        value = textOutput("prob_text"),
        showcase = bs_icon("graph-up"),
        theme = "secondary"
      ),

      uiOutput("dynamic_status_box")
    ),

    card(
      card_header("Risk Assessment Note"),
      paste(
        "The results above are calculated using a stepwise logistic regression model.",
        "Probabilities over 15% are flagged as High Risk based on historical default rates."
      )
    )
  )

  server <- function(input, output) {

    prediction_res <- eventReactive(
      input$predict,
      {

        new_data <- data.frame(
          Age = input$age,
          Income = input$income,
          LoanAmount = input$loanAmt,
          CreditScore = input$creditScore,
          MonthsEmployed = input$monthsEmp,
          InterestRate = input$interest,

          Education = factor(
            input$education,
            levels = levels(data_clean$Education)
          ),

          EmploymentType = factor(
            input$employment,
            levels = levels(data_clean$EmploymentType)
          ),

          # Fixed internal values for predictors retained by the model
          # but not exposed in the current demonstration UI.
          NumCreditLines = 2,
          DTIRatio = 0.3,

          MaritalStatus = factor(
            levels(data_clean$MaritalStatus)[1],
            levels = levels(data_clean$MaritalStatus)
          ),

          HasMortgage = factor(
            levels(data_clean$HasMortgage)[1],
            levels = levels(data_clean$HasMortgage)
          ),

          HasDependents = factor(
            levels(data_clean$HasDependents)[1],
            levels = levels(data_clean$HasDependents)
          ),

          LoanPurpose = factor(
            levels(data_clean$LoanPurpose)[1],
            levels = levels(data_clean$LoanPurpose)
          ),

          HasCoSigner = factor(
            levels(data_clean$HasCoSigner)[1],
            levels = levels(data_clean$HasCoSigner)
          ),

          # LoanTerm is retained here for exact alignment with the current app
          # input construction even though the stepwise model drops LoanTerm.
          LoanTerm = 36
        )

        prob <- predict(
          model_stepwise,
          newdata = new_data,
          type = "response"
        )

        return(
          round(prob * 100, 2)
        )
      }
    )

    output$prob_text <- renderText({
      paste0(
        prediction_res(),
        "%"
      )
    })

    output$dynamic_status_box <- renderUI({

      res <- prediction_res()

      # Demonstration threshold only; NOT a validated production cutoff.
      is_high <- res > 15

      value_box(
        title = "Risk Assessment",

        value = if (is_high) {
          "High Risk"
        } else {
          "Low Risk"
        },

        showcase = if (is_high) {
          bs_icon("shield-exclamation")
        } else {
          bs_icon("shield-check")
        },

        theme = if (is_high) {
          "danger"
        } else {
          "success"
        }
      )
    })
  }

  shinyApp(
    ui = ui,
    server = server
  )
}


# =============================================================================
# 21. COMPLETION MESSAGE
# =============================================================================

cat(
  "\n============================================================\n",
  "THESIS REPRODUCTION COMPLETE\n",
  "============================================================\n",
  "Outputs written to: ", normalizePath(OUTPUT_DIR), "\n",
  "Full data: 255,347 rows x 18 columns\n",
  "Development sample: 50,000 rows\n",
  "Train / holdout: ", nrow(train_data), " / ", nrow(test_data), "\n",
  "Stepwise predictors: ", length(selected_predictors), " (LoanTerm removed)\n",
  "Confusion matrix @ 0.50: TN=", TN,
  ", FN=", FN,
  ", FP=", FP,
  ", TP=", TP, "\n",
  "Accuracy: ", sprintf("%.4f", accuracy), "\n",
  "Default recall: ", sprintf("%.4f", default_recall), "\n",
  "Specificity: ", sprintf("%.4f", specificity), "\n",
  "Balanced accuracy: ", sprintf("%.4f", balanced_accuracy), "\n",
  "ROC-AUC: ", sprintf("%.3f", roc_auc), "\n",
  "All numerical thesis assertions passed.\n",
  "============================================================\n",
  sep = ""
)
