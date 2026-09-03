"""Apziva Project 1:
 Happy Customers.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Using one random state makes my results reproducible.
RANDOM_STATE = 42


# The script is in src, so parents[1] gives the main project folder.
project_folder = Path(__file__).resolve().parents[1]

data_file = project_folder / "data" / "raw" / "ACME-HappinessSurvey2020.xlsx"
figures_folder = project_folder / "reports" / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)
reports_folder = project_folder / "reports"
models_folder = project_folder / "models"
models_folder.mkdir(parents=True, exist_ok=True)

print("Project folder:", project_folder)
print("Looking for data at:", data_file)
print("File exists:", data_file.exists())


features = ["X1", "X2", "X3", "X4", "X5", "X6"]
question_names = {
    "X1": "Order delivered on time",
    "X2": "Contents were as expected",
    "X3": "Ordered everything wanted",
    "X4": "Paid a good price",
    "X5": "Satisfied with courier",
    "X6": "App makes ordering easy",
}


# 1. Load and explore the data
data = pd.read_excel(data_file)

print("First five rows:")
print(data.head())
print(f"\nDataset shape: {data.shape}")
print(f"Missing values: {data.isna().sum().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")
print("\nTarget counts (0 = unhappy, 1 = happy):")
print(data["Y"].value_counts().sort_index())
print("\nSurvey answer summary:")
print(data[features].describe().round(2))

# These checks help catch a data problem before modelling.
if list(data.columns) != ["Y"] + features:
    raise ValueError("The file should contain Y followed by X1 to X6.")
if data.isna().any().any():
    raise ValueError("The dataset contains missing values.")
if not data["Y"].isin([0, 1]).all():
    raise ValueError("Y should contain only 0 and 1.")
if not data[features].apply(lambda column: column.between(1, 5)).all().all():
    raise ValueError("Answers in X1 to X6 should be between 1 and 5.")

# I do not automatically remove identical rows. Different customers can
# genuinely give the same answers and the same happiness label.
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=data, x="Y", hue="Y", legend=False, ax=axes[0])
axes[0].set_title("Customer happiness")
axes[0].set_xlabel("0 = unhappy, 1 = happy")
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0,
            fmt=".2f", ax=axes[1])
axes[1].set_title("Correlation matrix")
fig.tight_layout()
fig.savefig(figures_folder / "data_overview.png", dpi=200)
# plt.show()
plt.close(fig)


# 2. Prepare the training and test data
X = data[features]
y = data["Y"]

# I keep 20% aside for the final test. It is not used to compare or tune models.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)


# 3. Compare suitable classification models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE),
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        SVC(class_weight="balanced", random_state=RANDOM_STATE),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# Repeated cross-validation is helpful because 126 observations is a small
# sample and the result from a single split can change considerably.
model_cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=RANDOM_STATE
)
model_results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    scores = cross_validate(
        model, X_train, y_train, cv=model_cv,
        scoring={"accuracy": "accuracy", "precision": "precision",
                 "recall": "recall", "f1": "f1"},
    )
    model_results.append({
        "Model": model_name,
        "Accuracy mean": scores["test_accuracy"].mean(),
        "Accuracy std": scores["test_accuracy"].std(),
        "Precision mean": scores["test_precision"].mean(),
        "Recall mean": scores["test_recall"].mean(),
        "F1 mean": scores["test_f1"].mean(),
        "F1 std": scores["test_f1"].std(),
    })

model_results = pd.DataFrame(model_results).sort_values(
    ["F1 mean", "Accuracy mean"], ascending=False
)
model_results.to_csv(reports_folder / "model_comparison.csv", index=False)
print("\nModel comparison using training data:")
print(model_results.round(3).to_string(index=False))


# 4. Tune the best model
# Gradient Boosting had the highest average F1 score in this comparison.
parameter_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.10],
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [1, 3, 5],
}
tuning_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    parameter_grid, scoring="f1", cv=tuning_cv,
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
joblib.dump(best_model, models_folder / "gradient_boosting_model.joblib")

print("\nBest Gradient Boosting parameters:")
print(grid_search.best_params_)
print(f"Best tuning F1 score: {grid_search.best_score_:.3f}")


# 5. Evaluate once on the untouched test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, zero_division=0)
test_recall = recall_score(y_test, test_predictions, zero_division=0)
test_f1 = f1_score(y_test, test_predictions, zero_division=0)

print("\nFinal test results:")
print(f"Accuracy:  {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1 score:  {test_f1:.3f}")
print("\nClassification report:")
print(classification_report(
    y_test, test_predictions, target_names=["Unhappy", "Happy"],
    zero_division=0,
))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, test_predictions, display_labels=["Unhappy", "Happy"],
    cmap="Blues", ax=ax,
)
fig.tight_layout()
fig.savefig(figures_folder / "confusion_matrix.png", dpi=200)
# plt.show()
plt.close(fig)


# 6. Find which questions matter most
# Permutation importance shows how much the test F1 score falls when the values
# of one feature are shuffled.
importance = permutation_importance(
    best_model, X_test, y_test, scoring="f1",
    n_repeats=100, random_state=RANDOM_STATE,
)
importance_table = pd.DataFrame({
    "Feature": features,
    "Question": [question_names[feature] for feature in features],
    "Importance mean": importance.importances_mean,
    "Importance std": importance.importances_std,
}).sort_values("Importance mean", ascending=False)

importance_table.to_csv(reports_folder / "feature_importance.csv", index=False)
print("\nPermutation feature importance:")
print(importance_table.round(3).to_string(index=False))

plot_data = importance_table.sort_values("Importance mean")
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_data["Question"], plot_data["Importance mean"],
        xerr=plot_data["Importance std"], color="teal")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Permutation feature importance")
ax.set_xlabel("Decrease in test F1 score after shuffling")
fig.tight_layout()
fig.savefig(figures_folder / "feature_importance.png", dpi=200)
# plt.show()
plt.close(fig)


# 7. Test smaller groups of survey questions
# With only six features, it is practical to test all 63 non-empty combinations.
subset_cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=RANDOM_STATE
)
subset_results = []

print("\nTesting all feature subsets. This may take several minutes...")

for number_of_features in range(1, len(features) + 1):
    print(f"Testing subsets containing {number_of_features} feature(s)...")
    
    for subset in combinations(features, number_of_features):
        subset = list(subset)
        scores = cross_validate(
            clone(best_model), X_train[subset], y_train, cv=subset_cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
        )
        subset_results.append({
            "Number of features": number_of_features,
            "Features": ", ".join(subset),
            "CV accuracy": scores["test_accuracy"].mean(),
            "CV F1": scores["test_f1"].mean(),
            "CV F1 std": scores["test_f1"].std(),
        })

subset_results = pd.DataFrame(subset_results)
# F1 by itself may favour predicting most customers as happy. I therefore use
# the average of accuracy and F1 when choosing the compact subset.
subset_results["Combined score"] = (
    subset_results["CV accuracy"] + subset_results["CV F1"]
) / 2
subset_results = subset_results.sort_values(
    ["Combined score", "Number of features"], ascending=[False, True]
)
subset_results.to_csv(
    reports_folder / "feature_subset_results.csv", index=False
)

best_subset = subset_results.iloc[0]
selected_features = best_subset["Features"].split(", ")
questions_to_remove = sorted(set(features) - set(selected_features))

print("\nBest five feature subsets:")
print(subset_results.head().round(3).to_string(index=False))
print(f"\nRecommended features: {best_subset['Features']}")
print("Possible questions to remove: " + ", ".join(questions_to_remove))

# This is an initial finding, not a final business decision. I would collect
# more customer responses before permanently removing survey questions.
final_summary = pd.DataFrame([{
    "Selected model": "Gradient Boosting",
    "Test accuracy": test_accuracy,
    "Test precision": test_precision,
    "Test recall": test_recall,
    "Test F1": test_f1,
    "Recommended features": best_subset["Features"],
    "Subset CV accuracy": best_subset["CV accuracy"],
    "Subset CV F1": best_subset["CV F1"],
}])
final_summary.to_csv(reports_folder / "final_summary.csv", index=False)

print(f"\nTables were saved in: {reports_folder}")
print(f"Figures were saved in: {figures_folder}")
print(f"Model was saved in: {models_folder}")
