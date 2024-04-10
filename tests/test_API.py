"""
Test pour API
"""

import pytest


from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_read_main():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_get_ids():
    """# ecrire une docstring"""

    response = client.get("/ids/")

    assert response.status_code == 200
    ids = response.json()

    print(f"le dataset contient {len(ids.values())} ids")
    assert len(ids.values()) >= 1


def test_client_details(id=100002.0):
    """ """

    # data_client = identite_client(id)
    # assert "CODE_GENDER" in data_client.columns

    pass


def test_client_prediction(id=100002.0):

    # prediction, statut = load_prediction(id)
    # assert prediction > 0

    pass


def test_shap_value(id=100002.0):

    # shap_id = shap_value(id)
    # assert len(shap_id) > 0

    pass


def test_nb():

    # nb = nb_credits()
    # assert nb > 0

    pass
