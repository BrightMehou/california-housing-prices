import logging

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.datasets import fetch_california_housing

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Chargement des données
logger.info("Chargement du dataset California Housing...")
housing = fetch_california_housing(as_frame=True)
data = housing.data
data["MedHouseVal"] = housing.target * 100000
logger.info("Dataset chargé avec succès.")

# Expander pour afficher les données
with st.expander("🔍 Voir les données brutes"):
    st.dataframe(data)

st.markdown(
    """
Voici un résumé du dataset California Housing utilisé dans ce projet :

**Attributs :**
1. **MedInc** - Revenu médian dans le groupe de blocs
2. **HouseAge** - Âge médian des maisons dans le groupe de blocs
3. **AveRooms** - Nombre moyen de pièces par ménage
4. **AveBedrms** - Nombre moyen de chambres par ménage
5. **Population** - Population du groupe de blocs
6. **AveOccup** - Nombre moyen de membres par ménage
7. **Latitude** - Latitude du groupe de blocs
8. **Longitude** - Longitude du groupe de blocs

**Variable Cible :** Valeur médiane des maisons pour les districts de Californie (en centaines de milliers de dollars):
"""
)

st.dataframe(data.describe())

st.markdown(
    """
**Observations Notables**

**Valeurs Extrêmes:**

- AveRooms, AveBedrms et AveOccup montrent des valeurs maximales très élevées par rapport à leurs moyennes et médianes. Cela suggère quelques districts extrêmes.
- Les valeurs extrêmes peuvent fausser les statistiques descriptives et les modèles notament les modèles linéaires, il peut donc être nécessaire de les traiter spécifiquement (par exemple, en les limitant, en les transformant selon les objectifs de l'analyse).

**Asymétrie:**

- MedInc et MedHouseVal tendent à être asymétriques à droite (longue traîne à droite) dans ce dataset, ce qui signifie qu'il y a un petit nombre de districts avec des revenus et des valeurs immobilières très élevés.

**Répartition Géographique:**

- Les limites de latitude et de longitude confirment que ces points de données proviennent de différentes régions de Californie, couvrant des zones côtières, des régions intérieures et possiblement des zones montagneuses ou désertiques.

**Relations Potentielles:**

- En général, MedInc est positivement corrélé avec MedHouseVal.
- HouseAge pourrait avoir une relation plus complexe ou plus faible avec MedHouseVal, mais les quartiers plus anciens dans certaines zones prisées peuvent quand même avoir des valeurs élevées.
- Population et AveOccup peuvent affecter indirectement les valeurs immobilières (par exemple, densité, urbain vs. rural).
"""
)

corr_matrix = data.corr()
fig = ff.create_annotated_heatmap(
    z=np.around(corr_matrix.values, decimals=3, out=None),
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.columns.tolist(),
    colorscale="Viridis",
)
fig.update_layout(title="Matrice de corrélation")

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """

### 🔑 Synthèse des corrélations

#### 1. Corrélations entre caractéristiques

* **AveRooms & AveBedrms**
  Corrélation très forte (\~0.85). Logique : plus un district a de pièces, plus il a aussi de chambres.
  ⚠️ Redondance possible → attention à la multicolinéarité en régression.
* **Latitude & Longitude**
  Relation fortement négative. Cela reflète la géographie de la Californie (les données suivent une diagonale nord-ouest → sud-est).

#### 2. Corrélations avec la cible (**MedHouseVal**)

* **MedInc (revenu médian)**
  Corrélation la plus élevée (\~0.69). Les districts plus riches ont des maisons plus chères → prédicteur central.
* **AveRooms**
  Corrélation positive modérée : plus de pièces = maisons légèrement plus chères.
* **HouseAge**
  Corrélation faible et positive : les zones plus anciennes peuvent être prisées, mais l’effet reste limité.
* **Population, AveOccup, AveBedrms**
  Corrélations très faibles (|r| < 0.05). Ces variables, seules, n’expliquent presque rien de la valeur des maisons.

#### 3. Autres relations intéressantes

* **HouseAge & Population** : légère corrélation négative → zones plus peuplées = constructions plus récentes en moyenne.
* **Longitude & Population** : faible corrélation positive → vers l’intérieur (est), certaines zones métropolitaines plus denses.
* **Asymétrie** : certaines variables (MedInc, MedHouseVal) sont étirées à droite. Des transformations (log) peuvent améliorer les modèles linéaires.

#### 4. Enseignements pratiques

1. **Prioriser MedInc** : variable la plus prédictive, potentiellement enrichissable (polynômes, interactions).
2. **Gérer la multicolinéarité** : AveRooms et AveBedrms apportent la même information → réduire la dimension ou n’en garder qu’une.
3. **Ne pas négliger la géographie** : Latitude et Longitude capturent une structure spatiale claire → envisager une modélisation géospatiale.
4. **Attention aux valeurs extrêmes** : AveRooms et AveOccup ont des outliers qui peuvent biaiser des modèles sensibles.
"""
)

feature = st.selectbox("Choisissez une variable :", data.columns)
fig = px.histogram(data, x=feature, nbins=30, title=f"Histogramme de {feature}")
st.plotly_chart(fig, use_container_width=True)


st.markdown(
"""
Nous pouvons d'abord nous concentrer sur les caractéristiques dont les distributions seraient plus ou moins attendues.

Le revenu médian présente une distribution avec une longue queue. Cela signifie que les salaires des individus sont plus ou moins distribués normalement, mais qu'il existe certaines personnes ayant des salaires très élevés.

En ce qui concerne l'âge moyen des maisons, la distribution est plus ou moins uniforme.

La distribution de la cible présente également une longue queue. De plus, il existe un effet de seuil pour les maisons de grande valeur : toutes les maisons avec un prix supérieur à 5 se voient attribuer la valeur 5.

En se concentrant sur les pièces moyennes, les chambres moyennes, l'occupation moyenne et la population, l'étendue des données est importante, avec des intervalles presque invisibles pour les valeurs les plus élevées. Cela signifie qu'il y a des valeurs très élevées et peu fréquentes.

Jusqu'à présent, nous avons écarté la longitude et la latitude, qui contiennent des informations géographiques. En résumé, la combinaison de ces caractéristiques pourrait nous aider à déterminer s'il existe des emplacements associés à des maisons de grande valeur. En effet, nous pourrions créer un graphique de dispersion où les axes x et y représenteraient la latitude et la longitude, et où la taille et la couleur des cercles seraient liées à la valeur des maisons dans chaque district.
"""
)
st.subheader("🗺️ Répartition géographique des logements")

st.markdown(
    """
Nous pouvons d'abord nous concentrer sur les caractéristiques dont les distributions seraient plus ou moins attendues.

Le revenu médian présente une distribution avec une longue queue. Cela signifie que les salaires des individus sont plus ou moins distribués normalement, mais qu'il existe certaines personnes ayant des salaires très élevés.

En ce qui concerne l'âge moyen des maisons, la distribution est plus ou moins uniforme.

La distribution de la cible présente également une longue queue. De plus, il existe un effet de seuil pour les maisons de grande valeur : toutes les maisons avec un prix supérieur à 5 se voient attribuer la valeur 5.

En se concentrant sur les pièces moyennes, les chambres moyennes, l'occupation moyenne et la population, l'étendue des données est importante, avec des intervalles presque invisibles pour les valeurs les plus élevées. Cela signifie qu'il y a des valeurs très élevées et peu fréquentes.

Jusqu'à présent, nous avons écarté la longitude et la latitude, qui contiennent des informations géographiques. En résumé, la combinaison de ces caractéristiques pourrait nous aider à déterminer s'il existe des emplacements associés à des maisons de grande valeur. En effet, nous pourrions créer un graphique de dispersion où les axes x et y représenteraient la latitude et la longitude, et où la taille et la couleur des cercles seraient liées à la valeur des maisons dans chaque district.         
"""
)

st.subheader("🗺️ Répartition géographique des logements")
fig = px.scatter_mapbox(
    data,
    lat="Latitude",  # Latitude des points
    lon="Longitude",  # Longitude des points
    color="MedHouseVal",  # Couleur basée sur la valeur médiane des maisons
    size="MedHouseVal",  # Taille des points basée sur la population
    hover_data=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
    ],  # Informations supplémentaires au survol
    color_continuous_scale="Viridis",  # Échelle de couleur
    size_max=15,
    zoom=5,
    height=600,
    mapbox_style="open-street-map",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
Nous remarquons que les maisons de grande valeur se situent sur la côte, là où se trouvent les grandes villes de Californie : San Diego, Los Angeles, San Jose ou San Francisco. Nous pouvons réaliser une analyse finale en créant un diagramme en paires (pair plot) de toutes les caractéristiques et de la cible.
"""
)
cols = data.columns.tolist()

# Sélection par l'utilisateur
x_axis = st.selectbox("Choisir la variable X", cols, index=0)
y_axis = st.selectbox("Choisir la variable Y", cols, index=1)
color_var = st.selectbox("Choisir la variable de couleur (optionnel)", [None] + cols)

# Création du scatter plot interactif
if color_var:
    fig = px.scatter(
        data,
        x=x_axis,
        y=y_axis,
        color=color_var,
        color_continuous_scale="Viridis",
        height=600,
        width=800,
    )
else:
    fig = px.scatter(data, x=x_axis, y=y_axis, height=600, width=800)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.markdown(
    """
Bien qu'il soit toujours compliqué d'interpréter un diagramme en paires (pairplot) en raison de la grande quantité de données, nous pouvons en tirer quelques intuitions. Nous pouvons confirmer que certaines caractéristiques présentent des valeurs extrêmes. Nous pouvons également observer que le revenu médian est utile pour distinguer les maisons de grande valeur de celles de faible valeur.

Ainsi, lors de la création d'un modèle prédictif, nous pouvons nous attendre à ce que la longitude, la latitude et le revenu médian soient des caractéristiques utiles pour prédire la valeur médiane des maisons.
"""
)
