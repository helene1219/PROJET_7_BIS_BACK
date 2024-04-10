"""
Assets driectly loaded 
"""

import pandas as pd


with open("pickle/best_lgbm.pkl", "rb") as f:
    best_LGBM = pd.read_pickle(f)

with open("pickle/data.pkl", "rb") as f:
    data = pd.read_pickle(f)

with open("pickle/encodage.pkl", "rb") as f:
    encodage = pd.read_pickle(f)

with open("pickle/imputer.pkl", "rb") as f:
    imputer = pd.read_pickle(f)

with open("pickle/scale.pkl", "rb") as f:
    scale = pd.read_pickle(f)

with open("pickle/sampler.pkl", "rb") as f:
    sampler = pd.read_pickle(f)

with open("pickle/estimateur.pkl", "rb") as f:
    estimateur = pd.read_pickle(f)
