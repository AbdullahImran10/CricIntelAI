
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD TRAINING AND TESTING DATA
# ==========================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# ==========================================
# 2. INITIALIZE MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

# ==========================================
# 3. TRAIN MODEL
# ==========================================

print("\n========== MODEL TRAINING ==========")

# Create mirrored training examples
X_train_mirrored = -X_train
y_train_mirrored = 1 - y_train

# Combine original and mirrored data
X_train_augmented = pd.concat(
    [X_train, X_train_mirrored],
    ignore_index=True
)

y_train_augmented = pd.concat(
    [y_train, y_train_mirrored],
    ignore_index=True
)

# Train model
model.fit(X_train_augmented, y_train_augmented.values.ravel())
print("Logistic Regression trained successfully!")

# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 5. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. SAVE MODEL
# ==========================================

joblib.dump(model, "models/logistic_regression.pkl")

print("\nModel saved successfully!")
print("Location: models/logistic_regression.pkl")
