
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

model.fit(X_train, y_train)

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
