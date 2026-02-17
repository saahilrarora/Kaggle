# Titanic Survival Prediction

Baseline model for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) — predict which passengers survived the sinking of the Titanic.

## Approach

We use a **Logistic Regression** classifier with manual feature engineering.

### Preprocessing

- **Title extraction** — Pull titles (Mr, Mrs, Miss, Master) from passenger names and group rare titles together.
- **Family size** — Combine `SibSp` and `Parch` into a single `FamilySize` feature.
- **Missing values** — Impute `Age` and `Fare` with training-set medians, `Embarked` with the training-set mode.
- **Encoding** — Map `Sex` to binary, one-hot encode `Embarked`.
- **Dropped columns** — `PassengerId`, `Name`, `Ticket`, `Cabin` (too many missing values), `SibSp`, `Parch` (replaced by `FamilySize`).

All imputation values are computed from the training set only to avoid data leakage.

### Model

- Logistic Regression (`max_iter=1000`, `random_state=42`)
- 5-fold cross-validation for performance estimation
- Expected accuracy: ~75–78%

## Usage

```bash
cd titanic
python titanic_baseline.py
```

Requires `train.csv` and `test.csv` from the Kaggle competition in the `titanic/` directory.

Outputs `titanic_submission.csv` ready for upload to Kaggle.

## Dependencies

- pandas
- numpy
- scikit-learn
