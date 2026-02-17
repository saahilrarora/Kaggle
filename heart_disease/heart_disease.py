import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

# Cross-validation
model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.2,
                      subsample=0.6, colsample_bytree=0.7, min_child_weight=3,
                      eval_metric="logloss", random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print(f"\nCross-Validation ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Fold scores: {[f'{s:.4f}' for s in scores]}")

# Train on full training set
model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print("=" * 45)
for _, row in importance.iterrows():
    bar = "#" * int(row["Importance"] * 100)
    print(f"  {row['Feature']:<30} {row['Importance']:.4f} {bar}")

# Predict probabilities on test set
predictions = model.predict_proba(test)[:, 1]

# Save submission
submission = pd.DataFrame({"id": test_ids, "Heart Disease": predictions})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: submission.csv ({len(submission)} rows)")
print(f"Sample predictions: {predictions[:5]}")
