"""
Test pour API
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_read_main():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_get_ids():
    """# teste sur liste Id"""

    response = client.get("/ids/")

    assert response.status_code == 200

    status_code = response.status_code
    content = response.json()

    # test dtype and shape
    assert isinstance(content, list)
    assert len(content) > 50


def test_features_id(client_id=100002.0):
    """# teste si ok récupération client_id selectionné"""

    response = client.get("/data_client/{client_id}")

    assert response.status_code == 200

    content = response.json()

    assert isinstance(content, dict)
    assert len(content) > 10


# def test_client_prediction(id=100007.0):
# """ vérifier prédiction du cient """

# response = client.get("/prediction/{client_id}")

# assert response.status_code == 200
# statut  = response.json()

# assert (statut := 0) | (statut := 1)


def test_shap_value(client_id=100002):

    response = client.get("/shap/{client_id}")

    assert response.status_code == 200
    shap_id = response.json()

    assert shap_id > 0
    # assert len(shap_id.values()) > 1


def test_nb():

    response = client.get("/nb_credit/")

    assert response.status_code == 200
    nb = response.json()

    assert nb > 0
