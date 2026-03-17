"""
Suite de tests Pytest pour l’application Streamlit et la fonction `model_prediction`.
Elle couvre :
- les prédictions réussies et les erreurs (exception, HTTP),
- l’état initial de l’interface Streamlit,
- et la validation d’une prédiction avec des entrées valides.
"""

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.ui.app import model_prediction


# Classe personnalisée pour être la valeur de retour mock
class MockResponseSuccess:
    """
    Représente une réponse simulée réussie pour le modèle.
    """

    status_code = 200

    @staticmethod
    def json():
        """
        Retourne une réponse simulée avec une prédiction.
        """
        return {
            "prediction": [3.0],
            "shap_values": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
        }


class MockResponseError:
    """
    Représente une réponse simulée avec une erreur HTTP.
    """

    status_code = 500
    text = "Erreur"

    @staticmethod
    def json():
        """
        Retourne un dictionnaire vide.
        """
        return {}


def test_model_prediction_success(monkeypatch) -> None:
    """
    Teste si `model_prediction` retourne correctement une prédiction
    lorsque le modèle répond avec succès.
    """

    def mock_post(*args, **kwargs) -> MockResponseSuccess:
        return MockResponseSuccess()

    monkeypatch.setattr(requests, "post", mock_post)

    input_data: dict[str, float] = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    text_output, shap_values = model_prediction(input=input_data)
    assert text_output == "💰 Le prix prédit pour le logement est : **300,000 $**."


def test_model_prediction_exception(monkeypatch) -> None:
    """
    Teste si `model_prediction` retourne un message d'erreur
    lorsqu'une exception est levée par la requête.
    """

    def mock_post(*args, **kwargs) -> requests.exceptions.RequestException:
        raise requests.exceptions.RequestException

    monkeypatch.setattr(requests, "post", mock_post)

    input_data = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    text_output, shap_values = model_prediction(input=input_data)
    assert text_output == "❌ Erreur : impossible de contacter le modèle."


def test_model_prediction_http_error(monkeypatch):
    """
    Teste si `model_prediction` retourne un message d'erreur
    lorsqu'une erreur HTTP est retournée par le modèle.
    """

    def mock_post(*args, **kwargs) -> MockResponseError:
        return MockResponseError()

    monkeypatch.setattr(requests, "post", mock_post)

    input_data = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    text_output, shap_values = model_prediction(input=input_data)
    assert text_output == "⚠️ Erreur : le modèle a retourné une réponse incorrecte."


@pytest.fixture
def session() -> AppTest:
    """
    Initialise une session de test Streamlit.
    """
    at = AppTest.from_file("src/ui/app.py")
    at.run(timeout=10)
    return at


def test_initial_state(session) -> None:
    """
    Vérifie l'état initial de l'interface Streamlit.

    - Vérifie le nombre de champs de texte et de boutons.
    - Vérifie les labels des champs et boutons.
    - Vérifie le nombre d'éléments Markdown.
    - Vérifie que les valeurs initiales des champs sont correctes.
    """

    # Vérification du nombre de champs number_input (8 champs attendus)
    assert len(session.number_input) == 8

    # Vérification du nombre de boutons (1 bouton attendu)
    assert len(session.button) == 1

    # Vérification des labels des champs number_input
    expected_labels: list[str] = [
        "Revenu médian des ménages (en dizaines de milliers de $)",
        "Âge moyen des maisons (en années)",
        "Nombre moyen de pièces par logement",
        "Nombre moyen de chambres par logement",
        "Population de la région",
        "Nombre moyen d'occupants par logement",
        "Latitude de la région",
        "Longitude de la région",
    ]
    for i, label in enumerate(expected_labels):
        assert session.number_input[i].label == label

    # Vérification des valeurs initiales des champs number_input
    # for field in session.number_input:
    #     assert field.value == 0.0  # Les valeurs initiales doivent être 0.0

    # Vérification du label du bouton
    assert session.button[0].label == "📈 Prédire"

    # Vérification du nombre d'éléments Markdown (3 attendus)
    assert len(session.markdown) == 3


def test_valid_input(session: AppTest, monkeypatch) -> None:
    """
    Vérifie que l'application Streamlit retourne une prédiction valide pour des entrées correctes.
    """

    def mock_post(*args, **kwargs) -> MockResponseSuccess:
        return MockResponseSuccess()

    monkeypatch.setattr(requests, "post", mock_post)

    # Fournir des entrées utilisateur valides (avec number_input)
    session.number_input[0].set_value(5.0).run()  # Revenu médian des ménages
    session.number_input[1].set_value(15.0).run()  # Âge moyen des maisons
    session.number_input[2].set_value(6.0).run()  # Nombre moyen de pièces par logement
    session.number_input[3].set_value(
        1.0
    ).run()  # Nombre moyen de chambres par logement
    session.number_input[4].set_value(800.0).run()  # Population de la région
    session.number_input[5].set_value(
        3.0
    ).run()  # Nombre moyen d'occupants par logement
    session.number_input[6].set_value(37.0).run()  # Latitude de la région
    session.number_input[7].set_value(-122.0).run()  # Longitude de la région

    # Cliquer sur "Prédire"
    session.button[0].click().run()

    # Vérification du message de résultat
    assert (
        session.success[0].value
        == "💰 Le prix prédit pour le logement est : **300,000 $**."
    )
