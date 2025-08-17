import logging
import os

import plotly.express as px
import requests
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(page_title="Prédiction du prix des logements", layout="wide")

logger.info("Démarrage de l'application Streamlit.")

# Titre principal
st.title("🏠 Prédiction du prix des logements en Californie")
st.markdown("""
Cette application utilise un modèle de machine learning pour prédire le prix des logements en Californie 
en fonction de plusieurs caractéristiques socio-démographiques et géographiques.
""")

# Chargement des données
logger.info("Chargement du dataset California Housing...")
housing = fetch_california_housing(as_frame=True)
data = housing.data
data["MedHouseVal"] = housing.target * 100000
logger.info("Dataset chargé avec succès.")

# Expander pour afficher les données
with st.expander("🔍 Voir les données brutes"):
    st.dataframe(data)

# Carte interactive avec Plotly
st.subheader("🗺️ Répartition géographique des logements")
logger.info("Affichage de la carte interactive.")

fig = px.scatter_mapbox(
    data,
    lat="Latitude",             # Latitude des points
    lon="Longitude",            # Longitude des points
    color="MedHouseVal",        # Couleur basée sur la valeur médiane des maisons
    size="MedHouseVal",          # Taille des points basée sur la population
    hover_data=["MedInc","HouseAge","AveRooms", "AveBedrms","Population","AveOccup"],  # Informations supplémentaires au survol
    color_continuous_scale="Viridis",     # Échelle de couleur
    size_max=15,
    zoom=5,
    height=600,
    mapbox_style="open-street-map",
)

st.plotly_chart(fig, use_container_width=True)

# URL du modèle
model_url = os.getenv("model_url", "http://localhost:8000/predict")
logger.info(f"URL du modèle : {model_url}")

# Fonction de prédiction
def model_prediction(input: dict):
    logger.info(f"Envoi des données au modèle : {input}")
    try:
        response = requests.post(model_url, json=input)
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de connexion au modèle : {e}")
        return "❌ Erreur : impossible de contacter le modèle.", None

    if response.status_code != 200:
        logger.error(f"Réponse invalide du modèle ({response.status_code}) : {response.text}")
        return "⚠️ Erreur : le modèle a retourné une réponse incorrecte.", None

    result = response.json()
    prediction = result["prediction"][0]
    shap_values = result.get("shap_values", [[]])[0]

    logger.info(f"Réponse reçue du modèle : {prediction} avec SHAP {shap_values}")

    text_output = f"💰 Le prix prédit pour le logement est : **{prediction*(10**5):,.0f} $**."
    return text_output, shap_values

# Formulaire utilisateur
st.subheader("🧾 Entrez les caractéristiques du logement")
col1, col2 = st.columns(2)
with col1:
    medinc = st.number_input("Revenu médian des ménages (en dizaines de milliers de $)", min_value=0.0, value=0.0)
    houseage = st.number_input("Âge moyen des maisons (en années)", min_value=0.0, value=0.0)
    averooms = st.number_input("Nombre moyen de pièces par logement", min_value=0.0, value=0.0)
    avebedrms = st.number_input("Nombre moyen de chambres par logement", min_value=0.0, value=0.0)
with col2:
    population = st.number_input("Population de la région", min_value=0.0, value=0.0)
    aveoccup = st.number_input("Nombre moyen d'occupants par logement", min_value=0.0, value=0.0)
    latitude = st.number_input("Latitude de la région", value=0.0)
    longitude = st.number_input("Longitude de la région", value=0.0)

bouton = st.button("📈 Prédire")
if bouton:
    input_data = {
        "medinc": medinc,
        "houseage": houseage,
        "averooms": averooms,
        "avebedrms": avebedrms,
        "population": population,
        "aveoccup": aveoccup,
        "latitude": latitude,
        "longitude": longitude,
    }
    logger.info("Formulaire soumis par l'utilisateur.")
    prediction_text, shap_values = model_prediction(input_data)

    if "Erreur" in prediction_text:
        st.error(prediction_text)
    else:
        st.success(prediction_text)

    if shap_values:
        feature_names = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
        shap_df = pd.DataFrame([shap_values], columns=feature_names)
        shap_df = shap_df.melt(var_name="Feature", value_name="SHAP value")

        fig = px.bar(shap_df, x="Feature", y="SHAP value", title="Importance des features (SHAP)")
        st.plotly_chart(fig)

st.markdown("---")
st.markdown("""
© 2025 - Application développée avec [Streamlit](https://streamlit.io/) | 
Construite avec **Python**, **Poetry**, **Scikit-learn**, **MLflow**, **FastAPI** et **Docker**.
""")

logger.info("Fin de chargement de la page.")
