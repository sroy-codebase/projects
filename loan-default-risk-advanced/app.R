# ==============================================================================
# STUDY 2 SHINY APP
# Frozen Full Logistic Regression + Frozen XGBoost Challenger
# Educational demonstration only — not for lending or credit decisions.
# ==============================================================================

required <- c("shiny", "bslib", "bsicons", "caret", "xgboost")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop(
    paste0(
      "Missing package(s): ", paste(missing, collapse = ", "),
      "\nInstall with: install.packages(c(",
      paste(sprintf('"%s"', missing), collapse = ", "), "))"
    )
  )
}

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(bsicons)
  library(caret)
  library(xgboost)
})

# ------------------------------------------------------------------------------
# 1. Load frozen model artifacts — NO training occurs here
# ------------------------------------------------------------------------------

# For deployment, place a COPY of the proven full Logistic model in models/.
# The second path is only a local-development fallback.
logistic_candidates <- c(
  file.path("models", "full_logistic_model.rds"),
  file.path("thesis_reproducibility_outputs", "models", "full_logistic_model.rds")
)

logistic_path <- logistic_candidates[file.exists(logistic_candidates)][1]
if (is.na(logistic_path)) {
  stop("Copy full_logistic_model.rds into models/full_logistic_model.rds")
}

paths <- c(
  xgb_model = file.path("models", "xgboost_final_model.rds"),
  preprocessor = file.path("models", "xgb_preprocessor.rds"),
  calibrator = file.path("models", "xgb_platt_calibrator.rds"),
  threshold = file.path("models", "xgb_research_threshold.rds"),
  config = file.path("models", "xgb_model_config.rds")
)

missing_files <- paths[!file.exists(paths)]
if (length(missing_files)) {
  stop(
    paste0(
      "Missing frozen model artifact(s):\n",
      paste0(" - ", missing_files, collapse = "\n")
    )
  )
}

logistic_model <- readRDS(logistic_path)
xgb_model <- readRDS(paths["xgb_model"])
xgb_preprocessor <- readRDS(paths["preprocessor"])
xgb_calibrator <- readRDS(paths["calibrator"])
xgb_threshold <- as.numeric(readRDS(paths["threshold"]))
xgb_config <- readRDS(paths["config"])

if (!inherits(logistic_model, "glm")) stop("Invalid Logistic model artifact.")
if (!inherits(xgb_model, "xgb.Booster")) stop("Invalid XGBoost model artifact.")
if (length(xgb_threshold) != 1 || !is.finite(xgb_threshold)) {
  stop("Invalid XGBoost threshold artifact.")
}

use_platt <- isTRUE(xgb_config$use_platt)
probability_variant <- if (use_platt) "Platt-calibrated XGBoost" else "Raw XGBoost"

# ------------------------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------------------------

safe_prob <- function(p, eps = 1e-7) {
  pmin(pmax(as.numeric(p), eps), 1 - eps)
}

safe_logit <- function(p, eps = 1e-7) {
  qlogis(safe_prob(p, eps))
}

model_factor <- function(name, value) {
  lv <- logistic_model$xlevels[[name]]
  if (is.null(lv) || !value %in% lv) stop(paste("Invalid factor level for", name))
  factor(value, levels = lv)
}

default_level <- function(name, preferred) {
  lv <- logistic_model$xlevels[[name]]
  if (preferred %in% lv) preferred else lv[1]
}

pretty_feature <- function(x) {
  labels <- c(
    Age = "Age", Income = "Income", LoanAmount = "Loan amount",
    CreditScore = "Credit score", MonthsEmployed = "Months employed",
    NumCreditLines = "Number of credit lines", InterestRate = "Interest rate",
    LoanTerm = "Loan term", DTIRatio = "Debt-to-income ratio"
  )
  if (x %in% names(labels)) return(unname(labels[x]))

  prefixes <- c(
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner"
  )
  for (p in prefixes) {
    if (startsWith(x, paste0(p, "."))) {
      return(paste0(p, ": ", sub(paste0("^", p, "\\."), "", x)))
    }
    if (startsWith(x, p) && x != p) {
      return(paste0(p, ": ", sub(paste0("^", p), "", x)))
    }
  }
  x
}

# ------------------------------------------------------------------------------
# 3. UI
# ------------------------------------------------------------------------------

ui <- page_sidebar(
  theme = bs_theme(bootswatch = "flatly"),
  title = "Explainable Loan Default Risk — Study 2",

  sidebar = sidebar(
    title = "Borrower & Loan Inputs",

    numericInput("age", "Age", 40, min = 18, max = 100),
    numericInput("income", "Annual Income ($)", 75000, min = 0, step = 1000),
    numericInput("loan_amount", "Loan Amount ($)", 100000, min = 0, step = 1000),
    numericInput("credit_score", "Credit Score", 650, min = 300, max = 850),
    numericInput("months_employed", "Months Employed", 36, min = 0),
    numericInput("interest_rate", "Interest Rate (%)", 10, min = 0, max = 100, step = 0.1),

    selectInput(
      "education", "Education",
      choices = logistic_model$xlevels$Education,
      selected = default_level("Education", "Bachelor's")
    ),

    selectInput(
      "employment_type", "Employment Type",
      choices = logistic_model$xlevels$EmploymentType,
      selected = default_level("EmploymentType", "Full-time")
    ),

    tags$details(
      tags$summary("Additional model inputs"),
      br(),
      numericInput("num_credit_lines", "Number of Credit Lines", 2, min = 0),
      numericInput("dti_ratio", "Debt-to-Income Ratio", 0.30, min = 0, max = 2, step = 0.01),

      selectInput(
        "marital_status", "Marital Status",
        choices = logistic_model$xlevels$MaritalStatus,
        selected = default_level("MaritalStatus", "Single")
      ),
      selectInput(
        "has_mortgage", "Has Mortgage?",
        choices = logistic_model$xlevels$HasMortgage,
        selected = default_level("HasMortgage", "No")
      ),
      selectInput(
        "has_dependents", "Has Dependents?",
        choices = logistic_model$xlevels$HasDependents,
        selected = default_level("HasDependents", "No")
      ),
      selectInput(
        "loan_purpose", "Loan Purpose",
        choices = logistic_model$xlevels$LoanPurpose,
        selected = default_level("LoanPurpose", "Other")
      ),
      selectInput(
        "has_cosigner", "Has Co-Signer?",
        choices = logistic_model$xlevels$HasCoSigner,
        selected = default_level("HasCoSigner", "No")
      ),
      selectInput(
        "loan_term", "Loan Term (months)",
        choices = c(12, 24, 36, 48, 60),
        selected = 36
      )
    ),

    hr(),
    actionButton("predict_btn", "Generate Assessment", class = "btn-primary w-100")
  ),

  card(
    card_header("Validated Research Context"),
    p(
      strong("XGBoost PR-AUC: 0.3189 vs Logistic: 0.3017. "),
      "Paired bootstrap 95% CI for the difference: 0.0063 to 0.0286."
    ),
    p(
      "ROC-AUC was 0.7553 vs 0.7513; the paired DeLong test was not significant ",
      "(p = 0.1396)."
    )
  ),

  layout_column_wrap(
    width = 1/3,
    value_box(
      title = "Logistic Probability",
      value = textOutput("logistic_pd"),
      showcase = bs_icon("graph-up"),
      theme = "secondary"
    ),
    value_box(
      title = "XGBoost Probability",
      value = textOutput("xgb_pd"),
      showcase = bs_icon("cpu"),
      theme = "primary"
    ),
    uiOutput("risk_box")
  ),

  layout_column_wrap(
    width = 1/2,
    card(
      card_header("Local XGBoost Explanation — TreeSHAP"),
      p(
        class = "text-muted small",
        "Positive SHAP values push estimated default risk up; negative values push it down."
      ),
      tableOutput("shap_table")
    ),
    card(
      card_header("Interpretation"),
      uiOutput("interpretation")
    )
  ),

  card(
    card_header("Important Research Disclaimer"),
    p(
      "Educational demonstration only. This application must not be used to approve, deny, ",
      "price, or otherwise make a real lending or credit decision. The XGBoost operating ",
      "threshold was selected from development out-of-fold predictions for this research ",
      "dataset and is not a validated production underwriting threshold."
    )
  )
)

# ------------------------------------------------------------------------------
# 4. Server
# ------------------------------------------------------------------------------

server <- function(input, output) {

  result <- eventReactive(input$predict_btn, {

    validate(
      need(input$age >= 18 && input$age <= 100, "Check Age."),
      need(input$income >= 0, "Income cannot be negative."),
      need(input$loan_amount >= 0, "Loan amount cannot be negative."),
      need(input$credit_score >= 300 && input$credit_score <= 850, "Credit score must be 300–850."),
      need(input$months_employed >= 0, "Months employed cannot be negative."),
      need(input$num_credit_lines >= 0, "Credit lines cannot be negative."),
      need(input$interest_rate >= 0 && input$interest_rate <= 100, "Check interest rate."),
      need(input$dti_ratio >= 0 && input$dti_ratio <= 2, "Check DTI ratio.")
    )

    new_data <- data.frame(
      Age = as.numeric(input$age),
      Income = as.numeric(input$income),
      LoanAmount = as.numeric(input$loan_amount),
      CreditScore = as.numeric(input$credit_score),
      MonthsEmployed = as.numeric(input$months_employed),
      NumCreditLines = as.numeric(input$num_credit_lines),
      InterestRate = as.numeric(input$interest_rate),
      LoanTerm = as.numeric(input$loan_term),
      DTIRatio = as.numeric(input$dti_ratio),
      Education = model_factor("Education", input$education),
      EmploymentType = model_factor("EmploymentType", input$employment_type),
      MaritalStatus = model_factor("MaritalStatus", input$marital_status),
      HasMortgage = model_factor("HasMortgage", input$has_mortgage),
      HasDependents = model_factor("HasDependents", input$has_dependents),
      LoanPurpose = model_factor("LoanPurpose", input$loan_purpose),
      HasCoSigner = model_factor("HasCoSigner", input$has_cosigner)
    )

    # Frozen full Logistic model
    logistic_pd <- safe_prob(
      predict(logistic_model, newdata = new_data, type = "response")
    )

    # Frozen XGBoost preprocessing + model
    xmat <- as.matrix(predict(xgb_preprocessor, newdata = new_data))
    storage.mode(xmat) <- "double"

    if (!is.null(xgb_config$feature_names) &&
        !identical(colnames(xmat), xgb_config$feature_names)) {
      stop("XGBoost feature schema mismatch. Use artifacts from the same final run.")
    }

    dnew <- xgb.DMatrix(xmat)
    raw_xgb_pd <- safe_prob(predict(xgb_model, dnew))

    # Apply the frozen calibration decision
    xgb_pd <- if (use_platt) {
      safe_prob(
        predict(
          xgb_calibrator,
          newdata = data.frame(oof_logit = safe_logit(raw_xgb_pd)),
          type = "response"
        )
      )
    } else {
      raw_xgb_pd
    }

    elevated <- xgb_pd >= xgb_threshold

    # Native XGBoost TreeSHAP for this borrower
    shap <- predict(xgb_model, dnew, predcontrib = TRUE)
    if (is.null(dim(shap))) shap <- matrix(shap, nrow = 1)
    shap <- as.matrix(shap)

    shap_values <- as.numeric(shap[1, seq_len(ncol(shap) - 1)])
    feature_names <- colnames(xmat)
    top_idx <- order(abs(shap_values), decreasing = TRUE)[
      seq_len(min(5L, length(shap_values)))
    ]

    shap_table <- data.frame(
      Feature = vapply(feature_names[top_idx], pretty_feature, character(1)),
      Direction = ifelse(
        shap_values[top_idx] > 0,
        "Higher risk",
        ifelse(shap_values[top_idx] < 0, "Lower risk", "Neutral")
      ),
      SHAP = round(shap_values[top_idx], 4),
      stringsAsFactors = FALSE
    )

    list(
      logistic_pd = logistic_pd,
      xgb_pd = xgb_pd,
      elevated = elevated,
      shap = shap_table
    )
  })

  output$logistic_pd <- renderText({
    if (input$predict_btn == 0) return("—")
    sprintf("%.1f%%", 100 * result()$logistic_pd)
  })

  output$xgb_pd <- renderText({
    if (input$predict_btn == 0) return("—")
    sprintf("%.1f%%", 100 * result()$xgb_pd)
  })

  output$risk_box <- renderUI({
    if (input$predict_btn == 0) {
      return(
        value_box(
          title = "Research Risk Flag",
          value = "Waiting",
          showcase = bs_icon("hourglass"),
          theme = "secondary"
        )
      )
    }

    r <- result()
    value_box(
      title = "Research Risk Flag",
      value = if (r$elevated) "Elevated Risk" else "Lower Risk",
      showcase = if (r$elevated) bs_icon("shield-exclamation") else bs_icon("shield-check"),
      theme = if (r$elevated) "danger" else "success",
      p(
        class = "small mb-0",
        paste0("XGBoost threshold: ", sprintf("%.1f%%", 100 * xgb_threshold))
      )
    )
  })

  output$shap_table <- renderTable({
    req(input$predict_btn > 0)
    result()$shap
  }, striped = TRUE, bordered = FALSE, hover = TRUE)

  output$interpretation <- renderUI({
    req(input$predict_btn > 0)
    r <- result()

    tagList(
      p(
        paste0(
          probability_variant, " estimates a probability of default of ",
          sprintf("%.1f%%", 100 * r$xgb_pd), ", which is ",
          if (r$elevated) "above" else "below", " the frozen research threshold of ",
          sprintf("%.1f%%", 100 * xgb_threshold), "."
        )
      ),
      p(
        paste0(
          "The frozen full Logistic Regression baseline estimates ",
          sprintf("%.1f%%", 100 * r$logistic_pd), " for the same borrower."
        )
      ),
      p(
        class = "text-muted small",
        "Both model probabilities are shown because the study validated XGBoost as a challenger; ",
        "the Logistic baseline was not altered. SHAP values are raw-margin contributions, not ",
        "percentage-point changes in probability."
      )
    )
  })
}

shinyApp(ui = ui, server = server)
