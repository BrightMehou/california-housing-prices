"""
Script de détection de dérive de données.
Ce fichier compare un jeu de données de référence à un jeu courant
et génère un rapport HTML indiquant les éventuelles dérives.
"""

import logging

from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_drift(
    reference_data: DataFrame,
    current_data: DataFrame,
    report_path: str = "data/data_drift_report.html",
) -> None:
    """
    Détecte la dérive entre deux jeux de données et génère un rapport HTML.

    Args:
        reference_data (DataFrame): Données de référence (ex. données d'entraînement).
        current_data (DataFrame): Données actuelles (ex. données de production).
        report_path (str): Chemin de sauvegarde du rapport HTML.
    """
    reference = Dataset.from_pandas(reference_data)
    current = Dataset.from_pandas(current_data)
    report = Report([DataDriftPreset()], include_tests=True)

    logger.info("🚦 Début de la détection de dérive...")
    my_eval = report.run(current, reference)
    my_eval.save_html(report_path)
    logger.info(f"📄 Rapport de dérive sauvegardé dans {report_path}")


if __name__ == "__main__":
    """
    Point d’entrée du script de détection de dérive.
    Simule les données de référence et actuelles, puis lance l’analyse.
    """
    logger.info("🔍 Chargement du dataset California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data["frame"]

    # Simulation des données
    train_data = df.sample(frac=0.7, random_state=42)
    prod_data = df.sample(frac=0.3, random_state=24)

    # Ajout de dérive artificielle
    prod_data["MedInc"] *= 1.2
    prod_data["AveOccup"] *= 1.7

    # Lancement de la détection
    detect_drift(train_data, prod_data, report_path="data/data_drift_report.html")
    logger.info("✅ Script terminé.")
