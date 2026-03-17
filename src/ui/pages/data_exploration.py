"""
Application Streamlit d’exploration du dataset California Housing.
Elle permet de visualiser les données brutes, d’analyser les statistiques descriptives,
d’explorer les corrélations entre variables, et de représenter les distributions
ainsi que la répartition géographique des logements en Californie.
"""

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

st.title("📊 Exploration des données - California Housing Dataset")
# Expander pour afficher les données
with st.expander("🔍 Voir les données brutes"):
    st.dataframe(data)

st.markdown(
    """
Voici un résumé du dataset California Housing utilisé dans ce projet :

**Caractéristiques :**
- **Instances :** 20 640
- **Attributs :** 8 attributs numériques et 1 variable cible
- **Valeurs Manquantes :** Aucune

**Attributs :**
1. **MedInc** - Revenu médian dans le groupe de blocs
2. **HouseAge** - Âge médian des maisons dans le groupe de blocs
3. **AveRooms** - Nombre moyen de pièces par ménage
4. **AveBedrms** - Nombre moyen de chambres par ménage
5. **Population** - Population du groupe de blocs
6. **AveOccup** - Nombre moyen de membres par ménage
7. **Latitude** - Latitude du groupe de blocs
8. **Longitude** - Longitude du groupe de blocs

**Variable Cible :** Valeur médiane des maisons pour les districts de Californie (en centaines de milliers de dollars)

**Source :** Dérivé du recensement américain de 1990, obtenu dans le dépôt StatLib. Il peut être chargé en utilisant la fonction `sklearn.datasets.fetch_california_housing`.

Le dataset représente des données au niveau du groupe de blocs, un groupe de blocs étant une petite unité géographique avec une population de 600 à 3 000 personnes. Des attributs comme le nombre moyen de pièces et de chambres par ménage peuvent donner des valeurs élevées dans les régions avec peu de ménages et beaucoup de maisons vides, telles que les stations de vacances.

Ce dataset a été référencé dans l'article suivant :
- Pace, R. Kelley et Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297

"""
)

st.subheader("📈 Statistiques descriptives")

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

st.subheader("Matrice de corrélation linéaire")
corr_matrix = data.corr()
fig = ff.create_annotated_heatmap(
    z=np.around(corr_matrix.values, decimals=3, out=None),
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.columns.tolist(),
    colorscale="Viridis",
)
fig.update_layout(title="Matrice de corrélation linéaire", height=700, width=700)

st.plotly_chart(fig, width='stretch')

st.markdown(
    """
### 1. Corrélations les plus fortes entre les caractéristiques

1. **AveRooms & AveBedrms**
   - C'est la corrélation la plus élevée entre les caractéristiques.
   - Intuitivement, si un district a un nombre moyen élevé de pièces, il aura généralement aussi un nombre moyen élevé de chambres.

2. **Latitude & Longitude**
   - Relation très fortement négative.
   - En Californie, lorsque la latitude augmente (vers le nord), la longitude devient souvent plus négative (vers l'intérieur ou l'ouest).
   - Cela est principalement dû à une caractéristique géographique : les données couvrent une bande diagonale de l'État (du nord-ouest au sud-est), entraînant une corrélation inverse forte.

---

### 2. Corrélation avec **MedHouseVal** (la cible)

- **MedInc & MedHouseVal**
  - C'est la relation la plus forte avec la cible.
  - En général, lorsque le revenu médian dans un district augmente, la valeur médiane des maisons augmente également.
  - Cela paraît logique : les zones plus riches ont tendance à avoir des valeurs immobilières plus élevées.

- **AveRooms & MedHouseVal**
  - Relation modérée et positive : les districts avec plus de pièces (en moyenne) tendent à avoir des valeurs immobilières légèrement plus élevées.

- **HouseAge & MedHouseVal**
  - Corrélation faible et positive.
  - Les districts plus anciens peuvent parfois se situer dans des zones prisées, bien que l'effet soit faible.

- **Population, AveOccup, AveBedrms**
  - Tous montrent une corrélation faible ou très faible avec MedHouseVal (tous près de \(|r| < 0.05\)).
  - Cela indique que ces caractéristiques, isolément, ne sont pas fortement prédictives de la valeur médiane des maisons dans un sens linéaire simple.

---

### 3. Autres observations notables

- **HouseAge & Population**
  - Corrélation négative faible à modérée : les zones avec une plus grande population pourraient avoir des développements immobiliers plus récents en moyenne, bien que la relation ne soit pas forte.

- **Longitude & Population**
  - Une légère corrélation positive suggère qu'à mesure que l'on se déplace vers l'est (longitude moins négative), la population pourrait être un peu plus grande, peut-être en raison de certaines zones métropolitaines intérieures.

- **Asymétrie vs. Corrélation**
  - Certaines caractéristiques comme **MedInc** et **MedHouseVal** peuvent être asymétriques à droite ; une corrélation élevée avec la cible peut être influencée par quelques districts à revenus ou valeurs très élevés.
  - Envisagez des transformations (par exemple, logarithmique) si les hypothèses du modèle linéaire sont importantes.

---

### 4. Enseignements pratiques

1. **Focalisez-vous sur le revenu médian** :
   - Sa corrélation de ~0.69 avec la valeur des maisons suggère qu'il s'agit d'un prédicteur crucial.
   - L'ingénierie des caractéristiques autour des revenus (par exemple, termes polynomiaux ou d'interaction) peut souvent améliorer les performances du modèle.

2. **Vérifiez les caractéristiques fortement corrélées** :
   - **AveRooms** et **AveBedrms** sont fortement corrélées entre elles (~0.85).
   - Dans un modèle de régression, cette multicolinéarité peut augmenter la variance dans les estimations des paramètres. On pourrait utiliser une réduction de dimensionnalité ou omettre l'une d'elles si elles véhiculent principalement les mêmes informations.

3. **Insights géospatiaux** :
   - La forte corrélation négative entre la latitude et la longitude suggère une distribution structurée et diagonale des données à travers la Californie.
   - Envisagez une modélisation cartographique ou géospatiale, car l'emplacement est souvent un facteur clé dans les prix de l'immobilier.

4. **Surveillez les valeurs extrêmes** :
   - Certaines variables (par exemple, **AveRooms**, **AveOccup**) présentent de fortes valeurs extrêmes. Même si la corrélation est faible, les valeurs extrêmes peuvent fortement influencer les résultats des modèles linéaires.

---

### Conclusion

La matrice de corrélation révèle que **le revenu médian** est de loin le prédicteur le plus fort de **la valeur médiane des maisons**, ce qui est cohérent avec l'intuition du monde réel que les districts plus riches tendent à avoir des prix immobiliers plus élevés. Les caractéristiques géospatiales (Latitude, Longitude) montrent une forte corrélation interne en raison de la géographie de la Californie. Pendant ce temps, **AveRooms** et **AveBedrms** sont fortement corrélées entre elles, indiquant une redondance potentielle dans la modélisation.

Ces insights soulignent l'importance de l'ingénierie des caractéristiques, de la gestion des valeurs extrêmes et éventuellement de l'analyse géospatiale lors de la construction de modèles prédictifs pour les valeurs immobilières en Californie.
"""
)

st.subheader("📊 Distribution des caractéristiques")
feature = st.selectbox("Choisissez une variable :", data.columns)
fig = px.histogram(data, x=feature, nbins=30, title=f"Histogramme de {feature}")
st.plotly_chart(fig, width='stretch')


st.markdown(
    """
Nous pouvons d'abord nous concentrer sur les caractéristiques dont les distributions seraient plus ou moins attendues.

Le revenu médian présente une distribution avec une longue queue. Cela signifie que les salaires des individus sont plus ou moins distribués normalement, mais qu'il existe certaines personnes ayant des salaires très élevés.

En ce qui concerne l'âge moyen des maisons, la distribution est plus ou moins uniforme.

La distribution de la cible présente également une longue queue. De plus, il existe un effet de seuil pour les maisons de grande valeur : toutes les maisons avec un prix supérieur à 5 se voient attribuer la valeur 5.

En se concentrant sur les pièces moyennes, les chambres moyennes, l'occupation moyenne et la population, l'étendue des données est importante, avec des intervalles presque invisibles pour les valeurs les plus élevées. Cela signifie qu'il y a des valeurs très élevées et peu fréquentes.

"""
)
st.subheader("🗺️ Répartition géographique des logements")

st.markdown(
    """
Jusqu'à présent, nous avons écarté la longitude et la latitude, qui contiennent des informations géographiques. En résumé, la combinaison de ces caractéristiques pourrait nous aider à déterminer s'il existe des emplacements associés à des maisons de grande valeur. En effet, nous pourrions créer un graphique de dispersion où les axes x et y représenteraient la latitude et la longitude, et où la taille et la couleur des cercles seraient liées à la valeur des maisons dans chaque district.         
"""
)

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

st.plotly_chart(fig, width='stretch')

st.markdown(
    """
Nous remarquons que les maisons de grande valeur se situent sur la côte, là où se trouvent les grandes villes de Californie : San Diego, Los Angeles, San Jose ou San Francisco. Nous pouvons réaliser une analyse finale en créant un diagramme en paires (pair plot) de toutes les caractéristiques et de la cible.
"""
)

st.subheader("📉 Diagramme en paires interactif")
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
    width='stretch',
)

st.markdown(
    """
Bien qu'il soit toujours compliqué d'interpréter un diagramme en paires (pairplot) en raison de la grande quantité de données, nous pouvons en tirer quelques intuitions. Nous pouvons confirmer que certaines caractéristiques présentent des valeurs extrêmes. Nous pouvons également observer que le revenu médian est utile pour distinguer les maisons de grande valeur de celles de faible valeur.

Ainsi, lors de la création d'un modèle prédictif, nous pouvons nous attendre à ce que la longitude, la latitude et le revenu médian soient des caractéristiques utiles pour prédire la valeur médiane des maisons.
"""
)
