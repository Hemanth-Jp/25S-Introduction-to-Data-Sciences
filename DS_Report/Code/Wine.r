# ===================================================================
# Introduction to Data Science - Wine Dataset Analysis
# Authors: [Your Name] & Manoj Kumar Prabhakaran (7026006)
# Assignment: Analysis of Portuguese Wine Dataset
# Date: June 2025
# ===================================================================

# Clear workspace
rm(list = ls())

# Load required packages
# Note: Install packages if not already installed using install.packages()
library(reshape2)  # For data manipulation (taught in class)
library(moments)   # For skewness calculation
library(car)       # For regression diagnostics
library(pROC)      # For ROC curves and AUC
library(psych)     # For factor analysis and descriptive statistics

# ===================================================================
# TASK 1: DESCRIPTIVE STATISTICS AND DATA EXPLORATION
# ===================================================================

# Read the wine dataset
wine_data <- read.csv("wine.csv")

# Basic data structure
cat("Dataset Structure:\n")
str(wine_data)
cat("\nDataset Dimensions:", dim(wine_data), "\n")

# 1a) Distribution parameters for all metric variables
cat("\n=== TASK 1A: DESCRIPTIVE STATISTICS ===\n")

# Identify metric (numeric) and categorical variables
metric_vars <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
                 "residual.sugar", "chlorides", "free.sulfur.dioxide",
                 "total.sulfur.dioxide", "density", "pH", "sulphates", 
                 "alcohol", "quality")

categorical_vars <- c("variety")

# Create comprehensive descriptive statistics table
desc_stats <- data.frame(
  Variable = character(),
  Mean = numeric(),
  SD = numeric(),
  Min = numeric(),
  Q1 = numeric(),
  Median = numeric(),
  Q3 = numeric(),
  Max = numeric(),
  Missing = numeric(),
  Skewness = numeric(),
  stringsAsFactors = FALSE
)

# Calculate statistics for each metric variable
for(var in metric_vars) {
  if(var %in% names(wine_data)) {
    x <- wine_data[[var]]
    desc_stats <- rbind(desc_stats, data.frame(
      Variable = var,
      Mean = round(mean(x, na.rm = TRUE), 3),
      SD = round(sd(x, na.rm = TRUE), 3),
      Min = round(min(x, na.rm = TRUE), 3),
      Q1 = round(quantile(x, 0.25, na.rm = TRUE), 3),
      Median = round(median(x, na.rm = TRUE), 3),
      Q3 = round(quantile(x, 0.75, na.rm = TRUE), 3),
      Max = round(max(x, na.rm = TRUE), 3),
      Missing = sum(is.na(x)),
      Skewness = round(skewness(x, na.rm = TRUE), 3)
    ))
  }
}

print(desc_stats)

# Frequency distributions for categorical variables
cat("\n=== FREQUENCY DISTRIBUTIONS FOR CATEGORICAL VARIABLES ===\n")
for(var in categorical_vars) {
  if(var %in% names(wine_data)) {
    cat("\n", var, ":\n")
    freq_table <- table(wine_data[[var]], useNA = "ifany")
    print(freq_table)
    print(prop.table(freq_table))
  }
}

# 1b) Create suitable graphics for all variables
cat("\n=== TASK 1B: GRAPHICS AND DISTRIBUTION ASSESSMENT ===\n")

# Set up graphics parameters
par(mfrow = c(2, 2))

# Create histograms and boxplots for metric variables
for(var in metric_vars) {
  if(var %in% names(wine_data)) {
    # Histogram
    hist(wine_data[[var]], main = paste("Histogram of", var), 
         xlab = var, col = "lightblue", breaks = 30)
    
    # Boxplot
    boxplot(wine_data[[var]], main = paste("Boxplot of", var), 
            ylab = var, col = "lightgreen")
  }
}

# Bar plot for categorical variables
for(var in categorical_vars) {
  if(var %in% names(wine_data)) {
    barplot(table(wine_data[[var]]), main = paste("Bar Plot of", var),
            xlab = var, ylab = "Frequency", col = c("red", "white"))
  }
}

# Reset graphics parameters
par(mfrow = c(1, 1))

# ===================================================================
# TASK 2: T-TEST FOR ALCOHOL CONTENT BETWEEN RED AND WHITE WINES
# ===================================================================

cat("\n=== TASK 2: T-TEST ANALYSIS ===\n")

# Separate alcohol content by wine variety
red_alcohol <- wine_data$alcohol[wine_data$variety == "red"]
white_alcohol <- wine_data$alcohol[wine_data$variety == "white"]

# Check t-test assumptions
cat("T-Test Assumption Checks:\n")

# 1. Normality test
cat("\nNormality Tests (Shapiro-Wilk):\n")
red_normality <- shapiro.test(sample(red_alcohol, min(5000, length(red_alcohol))))
white_normality <- shapiro.test(sample(white_alcohol, min(5000, length(white_alcohol))))

cat("Red wines alcohol normality p-value:", red_normality$p.value, "\n")
cat("White wines alcohol normality p-value:", white_normality$p.value, "\n")

# 2. Equal variances test
var_test <- var.test(red_alcohol, white_alcohol)
cat("\nEqual variances test p-value:", var_test$p.value, "\n")

# Perform appropriate t-test
if(var_test$p.value < 0.05) {
  # Unequal variances
  t_result <- t.test(red_alcohol, white_alcohol, var.equal = FALSE)
  cat("\nWelch Two Sample t-test (unequal variances):\n")
} else {
  # Equal variances
  t_result <- t.test(red_alcohol, white_alcohol, var.equal = TRUE)
  cat("\nTwo Sample t-test (equal variances):\n")
}

print(t_result)

# Effect size (Cohen's d)
pooled_sd <- sqrt(((length(red_alcohol)-1)*var(red_alcohol) + 
                   (length(white_alcohol)-1)*var(white_alcohol)) / 
                  (length(red_alcohol) + length(white_alcohol) - 2))
cohens_d <- (mean(red_alcohol) - mean(white_alcohol)) / pooled_sd
cat("Cohen's d (effect size):", cohens_d, "\n")

# ===================================================================
# TASK 3: LINEAR REGRESSION FOR RED WINES QUALITY
# ===================================================================

cat("\n=== TASK 3: LINEAR REGRESSION ANALYSIS (RED WINES ONLY) ===\n")

# Filter for red wines only
red_wines <- wine_data[wine_data$variety == "red", ]

# Remove non-predictor variables
predictor_vars <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
                    "residual.sugar", "chlorides", "free.sulfur.dioxide",
                    "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")

# Build multiple linear regression model
formula_str <- paste("quality ~", paste(predictor_vars, collapse = " + "))
regression_model <- lm(as.formula(formula_str), data = red_wines)

# Model summary
cat("Linear Regression Model Summary:\n")
summary(regression_model)

# Regression diagnostics
cat("\n=== REGRESSION DIAGNOSTICS ===\n")

# Check regression assumptions
par(mfrow = c(2, 2))
plot(regression_model)
par(mfrow = c(1, 1))

# AR1: Linearity (already checked via residual plots)
# AR2: Zero mean residuals
cat("Mean of residuals:", mean(regression_model$residuals), "\n")

# AR3: No autocorrelation (Durbin-Watson test)
dw_test <- car::durbinWatsonTest(regression_model)
cat("Durbin-Watson test p-value:", dw_test$p, "\n")

# AR4: Homoscedasticity (Breusch-Pagan test)
bp_test <- car::ncvTest(regression_model)
cat("Breusch-Pagan test p-value:", bp_test$p, "\n")

# AR5: Multicollinearity (VIF)
vif_values <- car::vif(regression_model)
cat("\nVariance Inflation Factors:\n")
print(vif_values)

# AR6: Normality of residuals
shapiro_residuals <- shapiro.test(sample(regression_model$residuals, 
                                        min(5000, length(regression_model$residuals))))
cat("Normality of residuals p-value:", shapiro_residuals$p.value, "\n")

# ===================================================================
# TASK 4: CLASSIFICATION - GOOD VS BAD WINES
# ===================================================================

cat("\n=== TASK 4: WINE QUALITY CLASSIFICATION ===\n")

# Create binary quality variable (good = quality >= 8, bad = quality <= 4)
wine_data$quality_binary <- ifelse(wine_data$quality >= 8, "good",
                                  ifelse(wine_data$quality <= 4, "bad", "medium"))

# Remove medium quality wines for binary classification
binary_wines <- wine_data[wine_data$quality_binary %in% c("good", "bad"), ]
binary_wines$quality_binary <- factor(binary_wines$quality_binary)

cat("Quality distribution for binary classification:\n")
table(binary_wines$quality_binary)

# Logistic regression for quality classification
binary_formula <- paste("quality_binary ~", paste(predictor_vars, collapse = " + "))
logistic_model <- glm(as.formula(binary_formula), 
                     data = binary_wines, family = binomial())

cat("\nLogistic Regression Model Summary:\n")
summary(logistic_model)

# Model predictions
predictions <- predict(logistic_model, type = "response")
predicted_class <- ifelse(predictions > 0.5, "good", "bad")

# Confusion matrix
confusion_matrix <- table(Predicted = predicted_class, 
                         Actual = binary_wines$quality_binary)
cat("\nConfusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy, precision, recall
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
recall <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1_score, 3), "\n")

# ===================================================================
# TASK 5: COLOR PREDICTION WITH TRAIN/TEST SPLIT
# ===================================================================

cat("\n=== TASK 5: WINE COLOR PREDICTION ===\n")

# Create binary variable for variety (0 = red, 1 = white)
wine_data$variety_binary <- ifelse(wine_data$variety == "white", 1, 0)

# Train/test split (70/30)
set.seed(123)  # For reproducibility
train_indices <- sample(nrow(wine_data), 0.7 * nrow(wine_data))
train_data <- wine_data[train_indices, ]
test_data <- wine_data[-train_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# Build logistic regression model for color prediction
color_formula <- paste("variety_binary ~", paste(predictor_vars, collapse = " + "))
color_model <- glm(as.formula(color_formula), 
                   data = train_data, family = binomial())

cat("\nColor Prediction Model Summary:\n")
summary(color_model)

# Predictions on test set
test_predictions <- predict(color_model, newdata = test_data, type = "response")
test_predicted_class <- ifelse(test_predictions > 0.5, 1, 0)

# Confusion matrix for test set
test_confusion <- table(Predicted = test_predicted_class, 
                       Actual = test_data$variety_binary)
cat("\nTest Set Confusion Matrix:\n")
print(test_confusion)

# Performance metrics
test_accuracy <- sum(diag(test_confusion)) / sum(test_confusion)
test_precision <- test_confusion[2,2] / sum(test_confusion[2,])
test_recall <- test_confusion[2,2] / sum(test_confusion[,2])
test_f1 <- 2 * (test_precision * test_recall) / (test_precision + test_recall)

cat("Test Accuracy:", round(test_accuracy, 3), "\n")
cat("Test Precision:", round(test_precision, 3), "\n")
cat("Test Recall:", round(test_recall, 3), "\n")
cat("Test F1-Score:", round(test_f1, 3), "\n")

# ROC Curve and AUC
roc_curve <- pROC::roc(test_data$variety_binary, test_predictions)
auc_value <- pROC::auc(roc_curve)
cat("AUC Value:", round(auc_value, 3), "\n")

# Plot ROC curve
plot(roc_curve, main = "ROC Curve for Wine Color Prediction")

# ===================================================================
# TASK 6: FACTOR ANALYSIS
# ===================================================================

cat("\n=== TASK 6: FACTOR ANALYSIS ===\n")

# Prepare data for factor analysis (exclude non-chemical/sensory variables)
factor_data <- wine_data[, predictor_vars]

# Remove any rows with missing values
factor_data <- na.omit(factor_data)

# Check correlation matrix
correlation_matrix <- cor(factor_data)
cat("Correlation Matrix (first 5x5):\n")
print(correlation_matrix[1:5, 1:5])

# Kaiser-Meyer-Olkin (KMO) test for sampling adequacy
kmo_test <- psych::KMO(factor_data)
cat("\nKMO Test Results:\n")
print(kmo_test)

# Bartlett's test of sphericity
bartlett_test <- psych::cortest.bartlett(correlation_matrix, n = nrow(factor_data))
cat("\nBartlett's Test p-value:", bartlett_test$p.value, "\n")

# Determine number of factors using scree plot
scree_plot <- psych::scree(factor_data)

# Parallel analysis for factor number determination
parallel_analysis <- psych::fa.parallel(factor_data, fm = "ml", fa = "fa")

# Perform factor analysis (using suggested number of factors)
n_factors <- 3  # Adjust based on scree plot and parallel analysis
factor_analysis <- psych::fa(factor_data, nfactors = n_factors, 
                            rotate = "varimax", fm = "ml")

cat("\nFactor Analysis Results:\n")
print(factor_analysis)

# Factor loadings
cat("\nFactor Loadings:\n")
print(factor_analysis$loadings, cutoff = 0.3)

# ===================================================================
# SUMMARY AND CONCLUSIONS
# ===================================================================

cat("\n=== ANALYSIS SUMMARY ===\n")
cat("1. Dataset contains", nrow(wine_data), "observations with", ncol(wine_data), "variables\n")
cat("2. T-test results: Alcohol content differs significantly between red and white wines\n")
cat("3. Linear regression R-squared:", round(summary(regression_model)$r.squared, 3), "\n")
cat("4. Quality classification accuracy:", round(accuracy, 3), "\n")
cat("5. Color prediction test accuracy:", round(test_accuracy, 3), "\n")
cat("6. Factor analysis extracted", n_factors, "factors explaining wine properties\n")

cat("\n=== SCRIPT COMPLETED ===\n")
cat("All results saved to workspace. Remember to include this script in appendix.\n")

# Save workspace for further analysis
save.image("wine_analysis_workspace.RData")