# Happy Customers: Customer Happiness Classification

## Project Overview

This project develops a machine learning model to predict whether a customer is happy or unhappy based on responses to six survey questions.

The target variable is:

- `Y = 0`: Unhappy customer
- `Y = 1`: Happy customer

The six input features represent customer ratings from 1 to 5.

The project also investigates which survey questions are most informative and whether a smaller subset of questions can retain useful predictive performance.

## Dataset

The dataset contains 126 customer responses and seven columns:

- `Y`: Customer happiness
- `X1`: Order delivered on time
- `X2`: Contents were as expected
- `X3`: Customer ordered everything they wanted
- `X4`: Customer paid a good price
- `X5`: Customer was satisfied with the courier
- `X6`: The app made ordering easy

The dataset contains no missing values.

Class distribution:

- 69 happy customers
- 57 unhappy customers

The dataset is stored locally under `data/raw/` and is excluded from version control.

## Project Workflow

The analysis includes:

1. Data validation and exploratory data analysis
2. Stratified 80/20 training and test split
3. Comparison of classification models
4. Repeated stratified cross-validation
5. Hyperparameter tuning using `GridSearchCV`
6. Evaluation on an untouched test set
7. Permutation feature importance
8. Evaluation of feature subsets

## Models Compared

The following classification models were evaluated:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting

Repeated stratified cross-validation was used to compare the models because the dataset is small and model performance may vary between different data splits.

## Final Model

Gradient Boosting achieved the highest mean cross-validation F1 score among the models compared and was selected for hyperparameter tuning.

The best Gradient Boosting parameters identified by `GridSearchCV` were:

- Learning rate: `0.01`
- Maximum tree depth: `2`
- Minimum samples per leaf: `1`
- Number of estimators: `50`

The best tuning cross-validation F1 score was `0.669`.

## Final Test Results

The tuned Gradient Boosting model achieved the following results on the untouched test set:

| Metric | Result |
|---|---:|
| Accuracy | 73.1% |
| Precision | 66.7% |
| Recall | 100% |
| F1 score | 0.800 |

The final test F1 score was therefore 80%.

Because the dataset contains only 126 observations, this test result should be interpreted together with the cross-validation results rather than as a standalone estimate of future performance.

## Feature Importance

Permutation importance indicated that the most influential features were:

- `X1`: Order delivered on time
- `X6`: App made ordering easy

`X3` showed a smaller contribution, while the remaining features had little effect on the test-set F1 score under the fitted model.

## Feature Selection

Feature-subset analysis indicated that the combination of `X1` and `X6` provided a strong compact feature set.

Its cross-validation results were:

- Accuracy: `0.598`
- F1 score: `0.717`

This suggests that delivery timeliness and ease of using the app may contain much of the predictive information in the survey.

The remaining questions may be candidates for removal. However, this is only an initial finding because the dataset is small. More customer responses should be collected before permanently changing the survey.

## Business Interpretation

The strongest signals in the analysis are related to:

- whether the order was delivered on time
- whether the app makes ordering easy

These results suggest that delivery reliability and ease of ordering may be especially important drivers of customer happiness.

However, the dataset is small, so these findings should be validated using additional customer data before making permanent business decisions.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/  
├── models/
├── notebooks/
│   └── happy_customers_analysis.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── feature_subset_results.csv
│   ├── final_summary.csv
│   └── figures/
└── src/
    ├── __init__.py
    └── happy_customers_analysis.py
