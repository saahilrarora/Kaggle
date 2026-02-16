"""
Titanic Survival Prediction - Baseline Logistic Regression Model
A simple classification model for the Kaggle Titanic competition
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the datasets
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Training set: {train.shape}")
print(f"Test set: {test.shape}")
print("\nMissing values in training set:")
print(train.isnull().sum())

# Save test PassengerIds for final submission
test_ids = test['PassengerId'].copy()

# ==========================================
# DATA PREPROCESSING
# ==========================================

print("\n" + "="*50)
print("PREPROCESSING DATA")
print("="*50)

# Calculate statistics from TRAINING set only (avoid data leakage)
median_age = train['Age'].median()
mode_embarked = train['Embarked'].mode()[0]
median_fare = train['Fare'].median()

print(f"\nImputation values (from training set):")
print(f"  - Age median: {median_age}")
print(f"  - Embarked mode: {mode_embarked}")
print(f"  - Fare median: {median_fare}")

def preprocess(df, median_age, mode_embarked, median_fare):
    """Apply identical preprocessing to a dataframe."""
    df = df.copy()

    # Fill missing values
    df['Age'] = df['Age'].fillna(median_age)
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['Fare'] = df['Fare'].fillna(median_fare)

    # Encode Sex: male=1, female=0
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-hot encode Embarked (C, Q, S → separate columns)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Drop columns we don't need
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

    return df

train = preprocess(train, median_age, mode_embarked, median_fare)
test = preprocess(test, median_age, mode_embarked, median_fare)

print("\nFinal feature columns:")
print(train.drop('Survived', axis=1).columns.tolist())

# ==========================================
# MODEL TRAINING
# ==========================================

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

# Separate features and target
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

print(f"\nTraining features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Cross-validation to estimate performance
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# GENERATE PREDICTIONS
# ==========================================

print("\n" + "="*50)
print("GENERATING PREDICTIONS")
print("="*50)

# Make predictions on test set
predictions = model.predict(test)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Survival rate in predictions: {predictions.mean():.2%}")

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

submission.to_csv('titanic_submission.csv', index=False)
print("\n✓ Submission file created: titanic_submission.csv")
print(f"  - Total predictions: {len(submission)}")
print(f"  - Survived: {(predictions == 1).sum()}")
print(f"  - Did not survive: {(predictions == 0).sum()}")

print("\n" + "="*50)
print("NEXT STEPS")
print("="*50)
print("1. Check titanic_submission.csv to verify the format")
print("2. Upload titanic_submission.csv to Kaggle")
print("3. Expected score: ~75-78% accuracy")
print("="*50)
