import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save test ids for submission, then drop id from both
test_ids = test["id"]
train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

# Encode target: Presence -> 1, Absence -> 0
train["Heart Disease"] = train["Heart Disease"].map({"Presence": 1, "Absence": 0})

# Split features and target
X_train = train.drop(columns=["Heart Disease"])
y_train = train["Heart Disease"]

# Hyperparameter search
param_distributions = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500, 700],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

model = XGBClassifier(eval_metric="logloss", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=50, cv=cv, scoring="accuracy",
    random_state=42, verbose=1, n_jobs=-1,
)

print("Starting hyperparameter search (50 iterations x 5 folds = 250 fits)...")
search.fit(X_train, y_train)

# Results
print(f"\nBest CV Accuracy: {search.best_score_:.4f}")
print(f"Best Parameters: {search.best_params_}")

results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
print("\nTop 10 Combinations:")
print("=" * 60)
for _, row in results.head(10).iterrows():
    print(f"  {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})  {row['params']}")

# Feature importance from best model
best_model = search.best_estimator_
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature Importance (best model):")
print("=" * 45)
for _, row in importance.iterrows():
    bar = "#" * int(row["Importance"] * 100)
    print(f"  {row['Feature']:<30} {row['Importance']:.4f} {bar}")

# Predict on test set using best model
predictions = best_model.predict(test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "Heart Disease": predictions})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: submission.csv ({len(submission)} rows)")
