from joblib import load
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model
model = load("model")

# Create the FastAPI app
app = FastAPI(title="Sample Project")

# Define input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Iris prediction model is running!"}

# Define endpoint
@app.post("/predict")
def predict(data: InputData):
    data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": str(prediction)}
