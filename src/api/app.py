"""
API FastAPI pour déployer un modèle MLflow de prédiction des prix des logements en Californie.
Elle charge le modèle et son explainer SHAP, puis expose des endpoints pour effectuer des prédictions
et interpréter les contributions des variables.
"""

from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

app = FastAPI(
    title="Prédiction des prix des logements en Californie",
    description="API simple pour prédire les prix des logements en Californie avec SHAP values",
    version="0.3.0",
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


def get_latest_run_id(model_name: str = "Production-model") -> str:
    """
    Retourne le run_id de la version la plus récente du modèle MLflow spécifié.

    Args:
        model_name (str): Nom du modèle MLflow.

    Returns:
        str: Identifiant de l'exécution (run_id).
    """
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version.run_id


RUN_ID: str = get_latest_run_id("Production-model")
MODEL_URI: str = f"runs:/{RUN_ID}/model"
EXPLAINER_URI: str = f"runs:/{RUN_ID}/explainer"

MODEL = mlflow.pyfunc.load_model(MODEL_URI)
EXPLAINER = mlflow.pyfunc.load_model(EXPLAINER_URI)


@app.get("/")
async def root() -> dict[str, str]:
    return {"msg": "API de prédiction des prix des logements opérationnelle ✅"}


@app.post("/predict")
def predict(input_data: InputFeatures) -> dict[str, list[Any]]:
    """
    Prédit le prix d’un logement en Californie à partir de ses caractéristiques.

    Cette fonction reçoit les données d’entrée sous forme d’un objet `InputFeatures`,
    les transforme en DataFrame compatible avec le modèle MLflow, puis retourne :
    - La prédiction du prix du logement
    - Les valeurs SHAP associées pour interpréter la contribution de chaque feature

    Paramètres :
    ----------
    input_data : InputFeatures
        Données d’entrée contenant les caractéristiques du logement :
        - medinc : revenu médian
        - houseage : âge moyen des habitations
        - averooms : nombre moyen de pièces
        - avebedrms : nombre moyen de chambres
        - population : population du quartier
        - aveoccup : taux d’occupation moyen
        - latitude : latitude géographique
        - longitude : longitude géographique

    Retour :
    -------
    dict
        Un dictionnaire contenant :
        - "prediction" : liste avec le prix prédit
        - "shap_values" : liste des valeurs SHAP pour chaque feature
    """

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
