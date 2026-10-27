import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(22, 60, n),
        "monthly_income": rng.integers(20000, 150000, n),
        "years_at_company": rng.integers(0, 20, n),
        "overtime": rng.integers(0, 2, n),
    })
    score = (df["overtime"] * 2 - df["years_at_company"] * 0.1
             - df["monthly_income"] / 50000 + rng.normal(0, 1, n))
    df["attrition"] = (score > score.median()).astype(int)
    return df


def train_model():
    df = make_data()
    X = df.drop(columns=["attrition"])
    y = df["attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/attrition_model.joblib")
    print("Model training completed. Accuracy:", round(accuracy, 3))
