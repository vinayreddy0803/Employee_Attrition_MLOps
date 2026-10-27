import subprocess
import sys
from pathlib import Path

from prefect import flow, task

ROOT = Path(__file__).resolve().parent.parent


@task
def prepare_project():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    print("Project prepared. Artifacts folder is ready.")
    return True


@task
def train_model(ready):
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    print(result.stdout.strip())
    return result.stdout.strip()


@task
def verify_results(training_output):
    model_file = ROOT / "artifacts" / "attrition_model.joblib"
    assert model_file.exists(), "Model file was not created"
    assert "Model training completed" in training_output
    print("Verification passed. Model file found:", model_file.name)


@flow(name="employee-attrition-pipeline")
def attrition_pipeline():
    ready = prepare_project()
    output = train_model(ready)
    verify_results(output)


if __name__ == "__main__":
    attrition_pipeline()
