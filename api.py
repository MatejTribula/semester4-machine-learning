import io

import mlflow
import pandas as pd
from fastapi import FastAPI, File, UploadFile

from data_standardization import get_inference_windows, preprocess

app = FastAPI()

model = mlflow.xgboost.load_model("models:/xgboost-classifier/Production")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    df = preprocess(df)

    results = []
    for user_id, date, X in get_inference_windows(df):
        if X is None:
            flare_up = None
        else:
            flare_up = int(model.predict(X.reshape(1, -1))[0])
        results.append({"user_id": user_id, "date": str(date), "flare_up": flare_up})

    return results

@app.get("/")
def index():
    return "Hello World"