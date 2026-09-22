import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("data/diabetes.csv")

# Features that users can realistically provide
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "BMI",
    "Age"
]

X = df[features]
y = df["Outcome"]

# --------------------------------
# 2. Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# 3. Create model
# --------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

# --------------------------------
# 4. Train
# --------------------------------

model.fit(X_train, y_train)

# --------------------------------
# 5. Evaluate
# --------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Diabetes Model Results =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------
# 6. Save model
# --------------------------------

joblib.dump(model, "models/diabetes_model_5features.joblib")

print("\nModel saved successfully!")
print("models/diabetes_model_5features.joblib")