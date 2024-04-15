"""
preprocess file 
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
import shap

from .assets import *




y = data["TARGET"]
X = data.drop(["SK_ID_CURR", "TARGET"], axis=1)
data_ID = data[["SK_ID_CURR"]]


cat_feat = X.select_dtypes(include=object).columns.to_list()
num_feat = X.select_dtypes(exclude=object).columns.to_list()

X_cat = X[cat_feat]
X_num = X[num_feat]


feature_names_out = encodage.get_feature_names_out(input_features=X_cat.columns)  # Obtenir les noms de colonnes étendus

X_cat_encoded = encodage.transform(X_cat)  # Transformer les données
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=feature_names_out, index=X.index)  # Convertir la matrice creuse en DataFrame

X_VF = pd.concat([X_num, X_cat_encoded_df], axis=1)
X_VF_Imputer = pd.DataFrame(imputer.fit_transform(X_VF), columns=X_VF.columns)
data_trans = pd.DataFrame(scale.fit_transform(X_VF_Imputer), columns=X_VF_Imputer.columns)
X_train_sample, y_train_sample = sampler.fit_resample(data_trans, y)


X_train_sample = X_train_sample.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
model_LGBM = estimateur.fit(X_train_sample, y_train_sample)
proba = model_LGBM.predict_proba(X_train_sample, pred_contrib=False)
proba_df = pd.DataFrame(proba, columns=["proba_classe_0", "proba_classe_1"])



shap_values_lgbm = model_LGBM.predict(X_train_sample, pred_contrib=True)


# shap-value

explainer = shap.TreeExplainer(model_LGBM)
shap_vals = explainer(X_train_sample)
