import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

app = FastAPI(
    title="Prédiction des prix des logements en Californie",
    description="API simple pour prédire les prix des logements en Californie avec SHAP values",
    version="0.1.0",
)


class InputFeatures(BaseModel):
    medinc: float
    houseage: float
    averooms: float
    avebedrms: float
    population: float
    aveoccup: float
    latitude: float
    longitude: float


def get_latest_run_id(model_name="Production-model"):
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version.run_id


RUN_ID = get_latest_run_id("Production-model")
MODEL_URI = f"runs:/{RUN_ID}/model"
EXPLAINER_URI = f"runs:/{RUN_ID}/explainer"

MODEL = mlflow.pyfunc.load_model(MODEL_URI)
EXPLAINER = mlflow.pyfunc.load_model(EXPLAINER_URI)


@app.get("/")
async def read_main():
    return {"msg": "API is running"}


@app.post("/predict")
def predict(input_data: InputFeatures):
    # Mapper les clés du JSON aux colonnes du modèle
    df = pd.DataFrame(
        [
            {
                "MedInc": input_data.medinc,
                "HouseAge": input_data.houseage,
                "AveRooms": input_data.averooms,
                "AveBedrms": input_data.avebedrms,
                "Population": input_data.population,
                "AveOccup": input_data.aveoccup,
                "Latitude": input_data.latitude,
                "Longitude": input_data.longitude,
            }
        ]
    )

    prediction = MODEL.predict(df)
    shap_values = EXPLAINER.predict(df)

    return {"prediction": prediction.tolist(), "shap_values": shap_values.tolist()}
