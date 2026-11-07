import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import make_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("employee-attrition")

df = make_data()
X = df.drop(columns=["attrition"])
y = df["attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_estimators in [10, 50, 100, 200]:
    with mlflow.start_run(run_name=f"rf-{n_estimators}-trees"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        print(f"n_estimators={n_estimators}  accuracy={acc:.3f}  f1={f1:.3f}")
