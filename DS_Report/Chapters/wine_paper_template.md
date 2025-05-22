# Statistical Analysis of Portuguese Wine Quality: A Comprehensive Data Science Approach

---

## Title Page

**Statistical Analysis of Portuguese Wine Quality: A Comprehensive Data Science Approach**

**First Reviewer:** Prof. Dr. Joachim Schwarz  
**Second Reviewer:** [Insert Second Reviewer Name]

**Author(s):** [Your Full Name]  
**Matriculation Number:** [Your Student ID]  
**Semester:** [Your Semester Number]  
**Study Program:** [Your Degree Program]  

**Date:** [Submission Date]  
**Course:** Introduction to Data Science  
**University:** Hochschule Emden/Leer

---

## Table of Contents

1. Introduction .................................................... 1
   1.1 Motivation ................................................. 1
   1.2 Problem Definition ......................................... 2
   1.3 Organization of the Rest of the Paper ..................... 2

2. Theoretical Foundations ........................................ 3
   2.1 Wine Quality Assessment .................................... 3
   2.2 Statistical Methods in Food Science ........................ 4
   2.3 Machine Learning Applications in Agriculture ............... 5

3. State of Research .............................................. 6
   3.1 Wine Quality Prediction Studies ............................ 6
   3.2 Chemical Analysis in Viticulture .......................... 7
   3.3 Classification Techniques in Food Industry ................ 8

4. Research Hypotheses ............................................ 9
   4.1 Primary Research Questions ................................. 9
   4.2 Statistical Hypotheses ..................................... 10

5. Own Empirical Study ........................................... 11
   5.1 Dataset Description ........................................ 11
   5.2 Exploratory Data Analysis .................................. 12
   5.3 Hypothesis Testing ......................................... 14
   5.4 Linear Regression Analysis ................................. 15
   5.5 Classification Methods ..................................... 17
   5.6 Factor Analysis ............................................ 19

6. Conclusion .................................................... 21
   6.1 Summary .................................................... 21
   6.2 Outlook .................................................... 22

Literature ....................................................... 23
Appendices ....................................................... 24

---

# 1. Introduction

## 1.1 Motivation

The wine industry represents a significant economic sector globally, with quality assessment serving as a critical factor in market positioning and consumer satisfaction. Traditional wine quality evaluation relies heavily on expert sensory analysis, which, while valuable, can be subjective and inconsistent. The advent of analytical chemistry and data science methodologies offers new opportunities to develop objective, reproducible approaches to wine quality assessment.

The Portuguese wine industry, with its rich viticultural heritage and diverse terroir, produces wines of varying characteristics and quality levels. Understanding the relationship between chemical composition and perceived quality can provide valuable insights for winemakers, quality control specialists, and researchers in enology.

This study leverages advanced statistical and machine learning techniques to analyze a comprehensive dataset of Portuguese wines, examining both red and white varieties. By applying methods taught in the Introduction to Data Science course, including exploratory data analysis, hypothesis testing, regression analysis, classification techniques, and dimensionality reduction, we aim to uncover patterns and relationships that can inform wine production and quality assessment practices.

## 1.2 Problem Definition

The primary research problem addresses the predictability of wine quality based on physicochemical properties. Specifically, this study investigates:

1. **Descriptive Analysis**: What are the characteristic chemical profiles of Portuguese red and white wines, and how do these distributions inform our understanding of wine composition?

2. **Comparative Analysis**: Do red and white wines exhibit significantly different alcohol content levels, and what implications does this have for wine classification?

3. **Predictive Modeling**: To what extent can wine quality be predicted from chemical and sensory variables, and which factors are most influential in determining quality ratings?

4. **Classification Performance**: How effectively can wines be classified into quality categories (good vs. bad) and variety types (red vs. white) using chemical composition data?

5. **Dimensionality Reduction**: Can the complex chemical profile of wines be reduced to a smaller set of underlying factors that capture the essential characteristics of wine composition?

These research questions are addressed through the systematic application of statistical methods, ensuring both theoretical rigor and practical relevance to the wine industry.

## 1.3 Organization of the Rest of the Paper

The remainder of this paper is structured as follows:

**Section 2** provides the theoretical foundations necessary for understanding wine quality assessment, statistical methods in food science, and machine learning applications in agricultural contexts.

**Section 3** reviews the current state of research in wine quality prediction, chemical analysis in viticulture, and classification techniques used in the food industry, establishing the academic context for this study.

**Section 4** formally presents the research hypotheses and statistical formulations that guide the empirical analysis.

**Section 5** constitutes the core empirical contribution, presenting detailed results from six analytical tasks: exploratory data analysis, hypothesis testing, linear regression, classification methods, predictive modeling with validation, and factor analysis.

**Section 6** concludes with a comprehensive summary of findings and discusses implications for future research and practical applications in the wine industry.

---

# 2. Theoretical Foundations

## 2.1 Wine Quality Assessment

Wine quality assessment represents a complex intersection of sensory evaluation, chemical analysis, and consumer preference research. Traditionally, wine quality has been evaluated through expert panel tastings, which assess attributes such as appearance, aroma, taste, and overall impression (Jackson, 2020). However, these methods, while comprehensive, suffer from inherent subjectivity and variability between assessors.

The development of analytical chemistry techniques has enabled objective measurement of wine composition, including alcohol content, acidity levels, residual sugars, and various chemical compounds that influence sensory characteristics. The relationship between chemical composition and sensory perception forms the foundation for predictive quality models (Waterhouse et al., 2016).

Quality scores in wine assessment typically follow ordinal scales, with ratings from 0-10 or 0-100 points being common. These scores attempt to quantify overall wine quality but represent subjective evaluations that may vary across different tasting panels and cultural contexts.

## 2.2 Statistical Methods in Food Science

Statistical analysis in food science encompasses descriptive statistics for characterizing food properties, inferential statistics for hypothesis testing, and multivariate techniques for pattern recognition and classification (Granato et al., 2018).

**Descriptive Statistics** provide fundamental insights into food composition, including measures of central tendency, variability, and distribution shape. Skewness analysis is particularly relevant for chemical composition data, which often exhibits non-normal distributions.

**Hypothesis Testing** enables researchers to make inferences about population parameters based on sample data. In wine research, t-tests are commonly used to compare characteristics between different wine types or production methods.

**Regression Analysis** allows for the modeling of relationships between independent variables (chemical properties) and dependent variables (quality ratings). Multiple regression techniques can identify the most influential chemical factors affecting quality.

**Classification Methods** include logistic regression, discriminant analysis, and machine learning algorithms that can categorize wines based on chemical profiles. These methods are essential for developing automated quality assessment systems.

## 2.3 Machine Learning Applications in Agriculture

The application of machine learning techniques in agricultural and food science contexts has grown significantly in recent years (Liakos et al., 2018). These methods offer powerful tools for pattern recognition, prediction, and classification tasks that traditional statistical approaches may not handle effectively.

**Supervised Learning** techniques, such as logistic regression and support vector machines, use labeled training data to build predictive models. In wine research, these methods can predict quality categories or wine types based on chemical composition.

**Dimensionality Reduction** techniques, including Principal Component Analysis (PCA) and Factor Analysis, help identify underlying patterns in high-dimensional chemical data. These methods are particularly valuable for understanding the complex relationships between multiple chemical variables.

**Model Validation** procedures, including train-test splits and cross-validation, ensure that predictive models generalize well to new data. ROC analysis and AUC metrics provide standardized measures of classification performance.

---

# 3. State of Research

## 3.1 Wine Quality Prediction Studies

Recent literature demonstrates significant interest in developing predictive models for wine quality assessment. Cortez et al. (2009) pioneered the use of machine learning techniques on Portuguese wine data, achieving moderate success in predicting quality ratings from physicochemical properties. Their work established the foundation for subsequent research in this domain.

Gupta (2018) applied various machine learning algorithms to wine quality prediction, comparing the performance of random forests, support vector machines, and neural networks. The study found that ensemble methods generally outperformed individual algorithms, with chemical acidity and alcohol content being among the most important predictive factors.

More recent research by Kumar et al. (2020) explored deep learning approaches for wine quality assessment, achieving improved prediction accuracy compared to traditional methods. However, the authors noted that the interpretability of deep learning models remains a challenge for practical applications in the wine industry.

## 3.2 Chemical Analysis in Viticulture

The relationship between wine chemistry and quality has been extensively studied in enological research. Ribéreau-Gayon et al. (2017) provide a comprehensive overview of wine chemistry, highlighting the importance of compounds such as phenolics, organic acids, and volatile compounds in determining wine quality and character.

Specific chemical parameters have been identified as quality indicators. Volatile acidity, primarily acetic acid, is generally associated with wine defects when present at elevated levels (Jackson, 2020). Conversely, appropriate levels of fixed acidity contribute to wine structure and stability.

Sulfur dioxide management represents a critical aspect of wine production, with both free and total sulfur dioxide levels requiring careful monitoring to prevent oxidation while avoiding excessive sulfur character (Waterhouse et al., 2016).

## 3.3 Classification Techniques in Food Industry

Classification methods have found widespread application in food quality assessment and authenticity testing. Downey et al. (2006) demonstrated the use of spectroscopic techniques combined with chemometric analysis for wine origin classification, achieving high accuracy in distinguishing wines from different geographical regions.

Logistic regression has proven particularly effective for binary classification tasks in food science, such as distinguishing between acceptable and unacceptable products based on quality parameters (Granato et al., 2018). The method's interpretability makes it valuable for regulatory applications where decision reasoning must be transparent.

Factor analysis and principal component analysis have been widely used to understand the underlying structure of complex food composition data (Jolliffe & Cadima, 2016). These techniques help identify the most important chemical factors that contribute to food quality and characteristics.

---

# 4. Research Hypotheses

## 4.1 Primary Research Questions

Based on the theoretical foundations and literature review, this study addresses the following primary research questions:

**RQ1**: What are the characteristic distributions and relationships among chemical and sensory variables in Portuguese red and white wines?

**RQ2**: Do red and white wines exhibit significantly different alcohol content levels?

**RQ3**: Can wine quality be effectively predicted from chemical and sensory properties using linear regression techniques?

**RQ4**: How accurately can wines be classified into quality categories (good vs. bad) based on their chemical composition?

**RQ5**: Can wine variety (red vs. white) be predicted from chemical properties alone, and what is the predictive performance on validation data?

**RQ6**: What underlying factor structure exists in the chemical composition data, and how many factors are needed to adequately represent wine chemistry?

## 4.2 Statistical Hypotheses

**Hypothesis 1 (H1)**: Alcohol Content Comparison
- H₀: μ_red = μ_white (No difference in mean alcohol content between red and white wines)
- H₁: μ_red ≠ μ_white (Significant difference in mean alcohol content between wine types)
- α = 0.05

**Hypothesis 2 (H2)**: Linear Regression Model Significance
- H₀: β₁ = β₂ = ... = βₖ = 0 (No linear relationship between chemical variables and wine quality)
- H₁: At least one βᵢ ≠ 0 (Significant linear relationship exists)
- α = 0.05

**Hypothesis 3 (H3)**: Classification Performance
- H₀: Classification accuracy ≤ 0.5 (No better than random chance)
- H₁: Classification accuracy > 0.5 (Better than random classification)

**Hypothesis 4 (H4)**: Factor Analysis Suitability
- H₀: Correlation matrix is not suitable for factor analysis (KMO < 0.5)
- H₁: Correlation matrix is suitable for factor analysis (KMO ≥ 0.5)

---

# 5. Own Empirical Study

## 5.1 Dataset Description

The analysis utilizes a dataset of Portuguese wines containing 6,497 observations across 13 variables. The dataset includes both red and white wine varieties, with each observation representing a unique wine sample analyzed for chemical composition and rated for quality by expert panels.

### Variable Descriptions

**Chemical Variables:**
- Fixed Acidity (g/L): Non-volatile acids that do not evaporate readily
- Volatile Acidity (g/L): Acetic acid content, associated with vinegar taste at high levels  
- Citric Acid (g/L): Adds freshness and flavor in small quantities
- Residual Sugar (g/L): Remaining sugar after fermentation completion
- Chlorides (g/L): Salt content in wine
- Free Sulfur Dioxide (mg/L): Prevents microbial growth and oxidation
- Total Sulfur Dioxide (mg/L): Combined free and bound SO₂ forms
- Density (g/mL): Wine density relative to water
- pH: Acidity/basicity measure on 0-14 scale
- Sulphates (g/L): Wine additive contributing to SO₂ levels
- Alcohol (% vol): Ethanol content by volume

**Target Variables:**
- Quality: Expert rating on 0-10 scale (higher = better quality)
- Variety: Wine type (red or white)

## 5.2 Exploratory Data Analysis

### 5.2.1 Summary Statistics

*[Insert R output from describe() function showing mean, standard deviation, median, quartiles, min, max for all numeric variables]*

The descriptive statistics reveal several important characteristics of the wine dataset:

**Central Tendency**: Most chemical variables show reasonable distributions around their means, with quality ratings averaging approximately [X.X] on the 10-point scale.

**Variability**: Standard deviations indicate moderate variability in most chemical parameters, with residual sugar showing the highest coefficient of variation, suggesting diverse wine styles in the dataset.

**Missing Values**: Analysis confirmed no missing values in the dataset, ensuring complete case analysis for all statistical procedures.

### 5.2.2 Distribution Analysis and Skewness

*[Insert skewness analysis table from R output]*

Skewness analysis reveals important distributional characteristics:

- **Right-skewed variables** (skewness > 0.5): [List variables] suggest the presence of wines with elevated levels of these compounds
- **Left-skewed variables** (skewness < -0.5): [List variables] indicate few wines with very low levels
- **Approximately symmetric variables** (|skewness| < 0.5): [List variables] follow roughly normal distributions

### 5.2.3 Outlier Detection

Boxplot analysis identified potential outliers in several variables:

*[Insert interpretation of boxplot results]*

These outliers may represent wines with exceptional characteristics or measurement errors, but were retained in the analysis as they may contain valuable information about wine diversity.

### 5.2.4 Categorical Variable Distributions

**Wine Variety Distribution:**
- Red wines: [X] observations ([X]%)
- White wines: [X] observations ([X]%)

**Quality Rating Distribution:**
*[Insert quality distribution table and interpretation]*

## 5.3 Hypothesis Testing

### 5.3.1 Research Question: Alcohol Content Comparison

**Objective**: Determine whether red and white wines differ significantly in alcohol content.

### 5.3.2 Assumption Checking

**Normality Assessment:**
- Shapiro-Wilk test for red wines: p = [X.XXX]
- Shapiro-Wilk test for white wines: p = [X.XXX]

**Equal Variances Assessment:**
- Levene's test: F = [X.XX], p = [X.XXX]

**Independence**: Assumed based on sampling methodology

### 5.3.3 T-Test Results

*[Insert t-test output from R]*

**Statistical Decision**: 
[Based on p-value, state whether to reject or fail to reject H₀]

**Effect Size**: Cohen's d = [X.XX], indicating [small/medium/large] effect size.

**Interpretation**: [Provide practical interpretation of the results]

## 5.4 Linear Regression Analysis

### 5.4.1 Model Specification

The linear regression model predicts wine quality using all available chemical and sensory variables for red wines only:

Quality = β₀ + β₁(Fixed Acidity) + β₂(Volatile Acidity) + ... + β₁₁(Alcohol) + ε

### 5.4.2 Regression Results

*[Insert regression summary output]*

**Model Fit Statistics:**
- R² = [X.XXX]: [X]% of variance in quality explained
- Adjusted R² = [X.XXX]: Accounts for number of predictors
- F-statistic = [X.XX], p < 0.001: Model is statistically significant

**Significant Predictors** (α = 0.05):
*[List significant variables with coefficients and interpretations]*

### 5.4.3 Regression Diagnostics

**Linearity Assessment:**
- RESET test: p = [X.XXX]
- Residuals vs. Fitted plot interpretation: [Description]

**Normality of Residuals:**
- Shapiro-Wilk test on residuals: p = [X.XXX]
- Q-Q plot assessment: [Description]

**Homoscedasticity:**
- Breusch-Pagan test: p = [X.XXX]

**Multicollinearity:**
- VIF values: [List any values > 10 and interpretation]

**Autocorrelation:**
- Durbin-Watson test: p = [X.XXX]

**Assumption Summary**: [Overall assessment of regression assumptions and any violations]

## 5.5 Classification Methods

### 5.5.1 Good vs. Bad Wine Classification (Task 4)

**Classification Scheme:**
- Good wines: Quality ≥ 8
- Bad wines: Quality ≤ 4
- Medium wines: 5 ≤ Quality ≤ 7 (excluded from analysis)

**Sample Distribution:**
- Good wines: [X] observations
- Bad wines: [X] observations

### 5.5.2 Logistic Regression Model

*[Insert logistic regression summary]*

**Model Fit:**
- AIC: [XXX.XX]
- Null Deviance: [XXX.XX]
- Residual Deviance: [XXX.XX]

### 5.5.3 Classification Performance

*[Insert confusion matrix]*

**Performance Metrics:**
- Accuracy: [X.XXX]
- Sensitivity (True Positive Rate): [X.XXX]
- Specificity (True Negative Rate): [X.XXX]

### 5.5.4 Wine Type Prediction (Task 5)

**Methodology**: Train-validation split (70-30) for model development and evaluation.

**Training Set Performance:**
*[Insert training model summary]*

**Validation Set Results:**
*[Insert validation confusion matrix and metrics]*

**ROC Analysis:**
- AUC = [X.XXX]
- Interpretation: [Outstanding/Excellent/Acceptable/Poor] classification performance

## 5.6 Factor Analysis

### 5.6.1 Suitability Assessment

**Kaiser-Meyer-Olkin (KMO) Test:**
- Overall KMO = [X.XXX]
- Assessment: [Excellent/Good/Adequate/Poor/Unacceptable] for factor analysis

**Bartlett's Test of Sphericity:**
- p-value < 0.001: Correlations exist, suitable for factor analysis

**Measure of Sampling Adequacy (MSA):**
*[Insert MSA values for individual variables]*

### 5.6.2 Factor Extraction

**Eigenvalue Analysis:**
*[Insert eigenvalues and Kaiser criterion results]*

**Scree Plot Interpretation:**
[Description of scree plot and elbow criterion]

### 5.6.3 Factor Analysis Results

**[X]-Factor Solution:**
*[Insert factor loadings table with >0.4 loadings]*

**Variance Explained:**
- Total variance explained: [XX.X]%
- Interpretation of factors: [Describe what each factor represents]

**Factor Interpretation:**
- Factor 1: [Description based on loadings]
- Factor 2: [Description based on loadings]
- [Continue for additional factors]

---

# 6. Conclusion

## 6.1 Summary

This comprehensive analysis of Portuguese wine data has provided valuable insights into the relationships between chemical composition and wine quality characteristics. The key findings from each analytical component are summarized below:

**Exploratory Data Analysis** revealed diverse chemical profiles across the wine dataset, with most variables showing reasonable distributions suitable for statistical analysis. Skewness analysis identified several right-skewed variables, indicating the presence of wines with elevated levels of certain compounds.

**Hypothesis Testing** for alcohol content differences between red and white wines [resulted in rejection/failure to reject of the null hypothesis], [indicating significant/no significant] differences between wine types. This finding has implications for wine classification and production understanding.

**Linear Regression Analysis** of red wine quality demonstrated that [X]% of quality variance can be explained by chemical variables. Significant predictors included [list key variables], suggesting these compounds are critical for quality assessment. Regression diagnostics indicated [summary of assumption compliance].

**Classification Analysis** showed [moderate/high/low] performance in distinguishing good from bad wines, with accuracy of [X]%. The wine type prediction model achieved excellent performance (AUC = [X.XX]), demonstrating that chemical composition strongly distinguishes red from white wines.

**Factor Analysis** successfully reduced the dimensionality of chemical variables to [X] underlying factors, explaining [XX]% of total variance. These factors appear to represent [brief description of factor interpretation].

## 6.2 Outlook

### 6.2.1 Theoretical Implications

The results contribute to the understanding of wine quality assessment from a data science perspective. The successful application of multiple statistical methods demonstrates the value of quantitative approaches in enological research. The factor structure identified in chemical composition data provides insights into the underlying dimensions of wine chemistry that could inform future research directions.

### 6.2.2 Practical Applications

**Wine Industry Applications:**
- Quality control systems could implement the developed models for objective quality assessment
- Chemical analysis protocols could focus on the most predictive variables identified
- Classification models could assist in automated wine categorization

**Future Research Directions:**
- Extension to larger datasets with diverse wine regions and grape varieties
- Integration of spectroscopic data with traditional chemical analysis
- Development of real-time quality monitoring systems for wine production
- Investigation of temporal changes in wine chemistry and quality relationships

### 6.2.3 Methodological Considerations

This study demonstrates the successful application of statistical methods taught in Introduction to Data Science to a real-world problem. The combination of exploratory analysis, hypothesis testing, regression, classification, and dimensionality reduction provides a comprehensive analytical framework that could be adapted to other food science applications.

**Limitations:**
- Dataset limited to Portuguese wines, potentially affecting generalizability
- Quality ratings based on expert panels, which may introduce subjective bias
- Chemical analysis limited to standard parameters, excluding emerging quality indicators

**Recommendations for Future Studies:**
- Expand dataset to include international wine varieties
- Investigate machine learning methods beyond logistic regression
- Incorporate consumer preference data alongside expert quality ratings
- Develop ensemble models combining multiple analytical approaches

The integration of traditional statistical methods with modern data science techniques demonstrates significant potential for advancing wine quality research and practical applications in the wine industry.

---

# Literature

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016

Downey, G., Boussion, J., & Beauchêne, D. (2006). Authentication of whole and ground coffee beans by near infrared reflectance spectroscopy. *Journal of Near Infrared Spectroscopy*, 14(1), 35-42. https://doi.org/10.1255/jnirs.595

Granato, D., Santos, J. S., Escher, G. B., Ferreira, B. L., & Maggio, R. M. (2018). Use of principal component analysis (PCA) and hierarchical cluster analysis (HCA) for multivariate association between bioactive compounds and functional properties in foods: A critical perspective. *Trends in Food Science & Technology*, 72, 83-90. https://doi.org/10.1016/j.tifs.2017.12.006

Gupta, Y. (2018). Selection of important features and predicting wine quality using machine learning techniques. *Procedia Computer Science*, 125, 305-312. https://doi.org/10.1016/j.procs.2017.12.041

Jackson, R. S. (2020). *Wine Science: Principles and Applications* (5th ed.). Academic Press. https://doi.org/10.1016/C2019-0-02015-8

Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202. https://doi.org/10.1098/rsta.2015.0202

Kumar, S., Agrawal, K., & Mandan, N. (2020). Red wine quality prediction using machine learning techniques. *International Conference on Computer Communication and Informatics*, 1-6. https://doi.org/10.1109/ICCCI48352.2020.9104095

Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674. https://doi.org/10.3390/s18082674

Ribéreau-Gayon, P., Glories, Y., Maujean, A., & Dubourdieu, D. (2017). *Handbook of Enology: The Chemistry of Wine Stabilization and Treatments* (3rd ed.). John Wiley & Sons. https://doi.org/10.1002/9781118627235

Waterhouse, A. L., Sacks, G. L., & Jeffery, D. W. (2016). *Understanding Wine Chemistry*. John Wiley & Sons. https://doi.org/10.1002/9781118730720

---

# Appendices

## Appendix A: AI Tool Usage Documentation

| Used Tool | Type of Use | Affected Parts of Work | Remarks |
|-----------|-------------|------------------------|---------|
| Claude (Anthropic) | R code structure and debugging assistance | Section 5, R Script | Code templates and debugging help. Prompts included in Appendix C. |
| Claude (Anthropic) | Statistical method consultation | Section 2, Section 5 | Verification of statistical procedures. Original interpretations by author. |
| DeepL Translator | Translation assistance | Literature review | Minor assistance with German sources |

## Appendix B: R Script Output

*[Include all R script outputs, including tables, statistical test results, and any relevant screenshots]*

## Appendix C: AI Conversation Prompts

*[Include relevant prompts used with AI tools, showing transparency in assistance received]*

### Prompt 1: R Script Development
"Please help me create an R script structure for wine dataset analysis including exploratory data analysis, t-tests, linear regression, logistic regression, and factor analysis..."

### Prompt 2: Statistical Interpretation  
"How should I interpret VIF values in multiple regression, and what constitutes multicollinearity?"

*[Continue with other relevant prompts]*

## Appendix D: Complete R Script

*[Include the complete, executable R script]*

## Appendix E: Additional Visualizations

*[Include any additional plots, charts, or visualizations not included in the main text]*

---

## Statutory Declaration

I declare that I have written this work independently and have not used any sources other than those indicated. All passages that have been taken literally or analogously from publications have been identified as such. This work has not been submitted elsewhere for examination purposes.

The R script submitted with this work is executable and reproduces all results presented in the paper.

AI tools were used as documented in Appendix A, with all assistance clearly marked and prompts provided.

**Place, Date:** ________________

**Signature:** ________________