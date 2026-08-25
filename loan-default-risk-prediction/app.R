# Loan default risk prediction using Logistic regression 
#Author : Soumen Roy (Data Scientist , MSDS)

library(tidyverse)
library(caret)
library(shiny)
library(bslib)
library(bsicons)

# 1. Load Data
loans_raw <- read_csv("Loan_default.csv", show_col_types = FALSE)

set.seed(1)
loan_data <- dplyr::sample_n(loans_raw, size = 50000)
rm(loans_raw)

# 2. Clean & Train Model
data_clean <- loan_data %>% 
  select(-LoanID) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(Default = as.factor(Default))

rm(loan_data)

set.seed(123)
trainIndex <- createDataPartition(data_clean$Default, p = 0.8, list = FALSE)
train_data <- data_clean[trainIndex, ]

full_model <- glm(Default ~ ., data = train_data, family = "binomial")
model_stepwise <- step(full_model, direction = "both", trace = 0)


#| label: shiny-ui

ui <- page_sidebar(
  theme = bs_theme(bootswatch = "flatly"),
  title = "Loan Default Risk Analyzer",
  sidebar = sidebar(
    title = "Borrower Inputs",
    numericInput("age", "Age", value = 35, min = 18),
    numericInput("income", "Annual Income ($)", value = 60000),
    numericInput("loanAmt", "Loan Amount ($)", value = 20000),
    numericInput("creditScore", "Credit Score", value = 700, min = 300, max = 850),
    numericInput("monthsEmp", "Months Employed", value = 48),
    numericInput("interest", "Interest Rate (%)", value = 6.5),
    selectInput("education", "Education Level", 
                choices = levels(data_clean$Education)),
    selectInput("employment", "Employment Type", 
                choices = levels(data_clean$EmploymentType)),
    hr(),
    actionButton("predict", "Generate Assessment", class = "btn-primary w-100")
  ),
  
  layout_column_wrap(
    width = 1/2,
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
    "The results above are calculated using a stepwise logistic regression model. 
     Probabilities over 15% are flagged as High Risk based on historical default rates."
  )
)

server <- function(input, output) {
  
  prediction_res <- eventReactive(input$predict, {
    new_data <- data.frame(
      Age = input$age,
      Income = input$income,
      LoanAmount = input$loanAmt,
      CreditScore = input$creditScore,
      MonthsEmployed = input$monthsEmp,
      InterestRate = input$interest,
      Education = factor(input$education, levels = levels(data_clean$Education)),
      EmploymentType = factor(input$employment, levels = levels(data_clean$EmploymentType)),
      NumCreditLines = 2,
      DTIRatio = 0.3,
      MaritalStatus = factor(levels(data_clean$MaritalStatus)[1], levels = levels(data_clean$MaritalStatus)),
      HasMortgage = factor(levels(data_clean$HasMortgage)[1], levels = levels(data_clean$HasMortgage)),
      HasDependents = factor(levels(data_clean$HasDependents)[1], levels = levels(data_clean$HasDependents)),
      LoanPurpose = factor(levels(data_clean$LoanPurpose)[1], levels = levels(data_clean$LoanPurpose)),
      HasCoSigner = factor(levels(data_clean$HasCoSigner)[1], levels = levels(data_clean$HasCoSigner)),
      LoanTerm = 36
    )
    
    prob <- predict(model_stepwise, newdata = new_data, type = "response")
    return(round(prob * 100, 2))
  })
  
  output$prob_text <- renderText({
    paste0(prediction_res(), "%")
  })
  
  output$dynamic_status_box <- renderUI({
    res <- prediction_res()
    is_high <- res > 15
    
    value_box(
      title = "Risk Assessment",
      value = if(is_high) "High Risk" else "Low Risk",
      showcase = if(is_high) bs_icon("shield-exclamation") else bs_icon("shield-check"),
      theme = if(is_high) "danger" else "success"
    )
  })
}

shinyApp(ui = ui, server = server)

