import io

import mlflow
import pandas as pd
from fastapi import FastAPI, File, UploadFile

from data_standardization import get_inference_windows, preprocess

app = FastAPI()


def load_production_model():
    client = mlflow.tracking.MlflowClient()
    for rm in client.search_registered_models():
        versions = client.get_latest_versions(rm.name, stages=["Production"])
        if versions:
            return mlflow.pyfunc.load_model(f"models:/{rm.name}/Production")
    raise RuntimeError("No model found in Production stage")


model = load_production_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    df = preprocess(df)

    results = []
    for user_id, date, X in get_inference_windows(df, window_size=5, predict_ahead=1):
        if X is None:
            symptom_degree = None
        else:
            symptom_degree = float(model.predict(pd.DataFrame([X]))[0])
        results.append({"user_id": user_id, "date": str(date), "symptom_degree": symptom_degree})

    return results


@app.get("/")
def index():
    return "Hello World"