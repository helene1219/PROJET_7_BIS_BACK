"""
preprocess file 
"""

import re

import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from .assets import *


data_ID = data[["SK_ID_CURR"]]
X=data.drop(["SK_ID_CURR"],axis=1)

cat_feat = X.select_dtypes(include=object).columns.to_list()
num_feat = X.select_dtypes(exclude=object).columns.to_list()

X_cat = X[cat_feat]
X_num = X[num_feat]


feature_names_out = encodage.get_feature_names_out(input_features=X_cat.columns)  # Obtenir les noms de colonnes étendus

X_cat_encoded = encodage.transform(X_cat)  # Transformer les données
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=feature_names_out, index=X.index)  # Convertir la matrice creuse en DataFrame

X_VF = pd.concat([X_num, X_cat_encoded_df], axis=1)
X_VF_Imputer = pd.DataFrame(imputer.fit_transform(X_VF), columns=X_VF.columns)
X_train_sample = pd.DataFrame(scale.fit_transform(X_VF_Imputer), columns=X_VF_Imputer.columns)

proba = estimateur.predict_proba(X_train_sample, pred_contrib=False)
proba_df = pd.DataFrame(proba, columns=["proba_classe_0", "proba_classe_1"])



# shap-value
shap_values_lgbm = estimateur.predict(X_train_sample, pred_contrib=True)
explainer = shap.TreeExplainer(estimateur)
shap_vals = explainer.shap_values(X_train_sample)
