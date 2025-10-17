import glob
import os

import streamlit as st
from PIL import Image

st.write(
    """
# 🧠 Analyse globale

La **dépendance partielle** montre la relation entre une ou plusieurs caractéristiques (*features*) 
et la prédiction moyenne du modèle, tout en "neutralisant" l'effet des autres caractéristiques.  
Elle répond à la question : *Comment les prédictions changent-elles en fonction d'une caractéristique donnée, en moyenne ?*

**Limites :**
- Suppose que les caractéristiques sont indépendantes, ce qui n'est pas toujours vrai dans les données réelles.
- Les interactions entre variables ne sont pas directement visibles.
"""
)

# Répertoire contenant les images
image_dir = "data/feature_analysis"
pattern = os.path.join(image_dir, "*.png")

# Récupération et tri des fichiers
image_files = sorted(glob.glob(pattern))

if not image_files:
    st.warning(f"⚠️ Aucune image trouvée dans le dossier '{image_dir}'.")
else:
    for filepath in image_files:
        image = Image.open(filepath)
        filename = os.path.basename(filepath).replace(".png", "")
        st.subheader(f"📈 {filename}")
        st.image(image, width="stretch")
        st.divider()
    st.write(
        """
    **MedInc (Revenu Médian par Groupe)**
On observe une corrélation positive entre le revenu médian et la valeur médiane des maisons. Plus le revenu médian augmente, plus la valeur moyenne des prédictions des maisons croît. Cela confirme les observations faites lors de l’analyse exploratoire des données.

---

**HouseAge (Âge des Maisons)**
La relation entre l’âge des maisons et leur valeur suit une courbe en forme de U. Les maisons neuves sont estimées avec une valeur plus élevée que celles ayant entre 5 et 20 ans. Cependant, à mesure que l'âge des maisons augmente au-delà de 20 ans, leur valeur moyenne tend à augmenter progressivement. Cela suggère qu'un âge avancé peut également être perçu comme une marque de caractère ou de localisation avantageuse.

---

**AveRooms (Nombre Moyen de Pièces par Logement)**
La valeur médiane des maisons augmente de manière significative avec le nombre moyen de pièces par logement, suivant des paliers bien distincts :
- Les logements ayant entre **1 et 3 pièces** sont en moyenne évalués à **200 000 dollars**.
- Ceux ayant entre **3 et 60 pièces** atteignent en moyenne **260 000 dollars**.
- Enfin, les logements avec plus de **60 pièces** sont estimés autour de **340 000 dollars**.  
Ces observations montrent que la taille des logements a un impact notable sur leur valeur.

---

**AveBedrms (Nombre Moyen de Chambres par Logement)**
La valeur médiane des maisons varie légèrement par paliers en fonction du nombre moyen de chambres :
- Les logements ayant entre **1 et 10 chambres** sont estimés en moyenne à **230 000 dollars**.
- Ceux ayant entre **10 et 17 chambres** voient leur valeur légèrement diminuer à **220 000 dollars**.
- Enfin, au-delà de **17 chambres**, la valeur augmente à nouveau pour atteindre **235 000 dollars**.  
Cette variation peut s’expliquer par des interactions complexes entre le nombre de chambres et d'autres caractéristiques, comme la surface totale.

---

**Population (Densité de Population)**
La densité de population n’a qu’un impact limité sur la valeur médiane des maisons. La valeur des biens augmente légèrement avec la densité de population, mais seulement jusqu’à un certain seuil, au-delà duquel elle stagne. Cela suggère que la densité seule n’est pas un facteur décisif dans la détermination des prix immobiliers.

---

**AveOccup (Nombre Moyen de Personnes par Logement)**
Le nombre moyen de personnes par logement semble avoir une influence très faible sur les prédictions. La valeur moyenne des biens reste presque constante, quel que soit ce nombre. Cela est logique, car la valeur immobilière dépend principalement de la localisation et des caractéristiques physiques des biens, plutôt que du nombre de personnes qui y résident.

---

**Latitude**
La valeur des biens immobiliers diminue de manière progressive à mesure que la latitude augmente. Cela confirme l’hypothèse formulée lors de l’analyse exploratoire : les biens situés plus au sud, proches des zones côtières, ont des valeurs plus élevées. Cela reflète la forte demande pour les zones proches de l’océan Pacifique.

---

**Longitude**
De manière similaire à la latitude, la valeur des biens diminue lorsque la longitude augmente, c’est-à-dire en s’éloignant de la côte californienne. Cette observation est cohérente avec les villes comme **San Diego**, **Los Angeles**, **San Jose**, ou **San Francisco**, où les prix immobiliers sont nettement plus élevés en raison de leur proximité avec la mer et leur attractivité économique.
    """
    )

st.write(
    """
# Analuse locale avec les valeurs SHAP
         
Définition :
Les valeurs SHAP mesurent la contribution exacte de chaque caractéristique à la prédiction d'une observation spécifique, en utilisant les valeurs de Shapley issues de la théorie des jeux.

Méthode :
Pour une observation donnée, les valeurs SHAP calculent la différence entre la prédiction obtenue avec et sans chaque caractéristique, en considérant toutes les combinaisons possibles.
Les valeurs SHAP sont additives : la somme des contributions de toutes les caractéristiques est égale à la prédiction finale moins la valeur de base (expected value).

Utilisation :
Explications locales (par observation) et globales (en agrégeant les valeurs SHAP).
Répondre à la question : Pourquoi ce modèle a-t-il fait cette prédiction pour cette observation ?
Exemple :
Pour une observation où la prédiction est élevée, les valeurs SHAP vous diront quelles caractéristiques ont poussé la prédiction à augmenter (et de combien), et lesquelles l'ont fait baisser.

Avantages :
Interactions : Les valeurs SHAP prennent en compte les interactions entre caractéristiques.
Applicables aux explications locales et globales.
"""
)
