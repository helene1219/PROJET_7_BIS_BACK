from API import get_ids
from API import identite_client
from API import load_prediction
from API import shap_value
from API import nb_credits

#import pandas as pd
#import logging

#logging.basicConfig(filename='test.log', level=logging.DEBUG) #, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S' )

def test_get_ids():
    ids = get_ids()
    print(f'le dataset contient {len(ids.values())} ids')
    assert len(ids.values()) >= 1


def test_client_details(id=100002.0):
    data_client = identite_client(id)
    assert 'CODE_GENDER' in data_client.columns
    
def test_client_prediction(id=100002.0):
    prediction,statut = load_prediction(id)
    assert prediction>0

    
def test_shap_value(id=100002.0):
    shap_id=shap_value(id)
    assert len(shap_id)>0
    
    
def test_nb():
    nb=nb_credits()
    assert nb>0