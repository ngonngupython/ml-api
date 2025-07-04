# cicd2_webservice.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.post("/predict")
def predict(input_data: IrisInput):
    data = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

# uvicorn cicd2_webservice:app --reload

# curl -X POST "http://127.0.0.1:8000/predict" \
#   -H "Content-Type: application/json" \
#   -d '{"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2}'