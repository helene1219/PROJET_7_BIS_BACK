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
    
    
def test_get_ids():
    """# teste sur liste Id"""

    response = client.get("/test_nb_colonne/")

    assert response.status_code == 200

    status_code = response.status_code
    content = response.json()

    # test dtype and shape
    assert content ==192


def test_features_id():
    """# teste si ok récupération client_id selectionné"""

    client_id = 100006
    response = client.get(f"/data_client/{client_id}")

    assert response.status_code == 200

    content = response.json()

    assert isinstance(content, dict)
    assert len(content) > 10


def test_client_prediction():
    """vérifier prédiction du cient"""

    client_id = 100002
    response = client.get(f"/prediction/{client_id}")

    assert response.status_code == 200
    statut = response.json()

    assert (statut := 0) | (statut := 1)


def test_shap_value():

    client_id = 100002
    response = client.get(f"/shap/{client_id}")

    assert response.status_code == 200
    shap_id = response.json()

    assert isinstance(shap_id, dict)
    #assert len(shap_id.values()) > 10


def test_nb():

    response = client.get("/nb_credit/")

    assert response.status_code == 200
    nb = response.json()

    assert nb > 0
