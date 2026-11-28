
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# Load training data only
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# Define model
model = LogisticRegression(max_iter=2000)

# Hyperparameters to test
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"]
}

# Time-aware cross-validation
time_split = TimeSeriesSplit(n_splits=5)

# Grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=time_split,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("\n========== HYPERPARAMETER TUNING ==========")

grid_search.fit(X_train, y_train)

# Display best results
print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Save best model
best_model = grid_search.best_estimator_

joblib.dump(
    best_model,
    "models/best_logistic_regression.pkl"
)

print("\nBest model saved successfully!")
