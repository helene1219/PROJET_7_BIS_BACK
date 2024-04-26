"""
routes file
"""

from fastapi import FastAPI, Response  # , UploadFile

from .preprocess import *

app = FastAPI()

def explanation_to_dict(single_exp: shap._explanation.Explanation) -> dict:
    """
    Convert Shap explanation to dict
    """
    v = pd.DataFrame(single_exp.values)[0].to_dict()
    d = pd.DataFrame(single_exp.data)[0].to_dict()
    dd = single_exp.display_data.to_dict()
    b = single_exp.base_values
    return {"values": v, "base_values": b, "data": d, "display_data": dd}


@app.get("/")
def read_main():

    return {"msg": "Hello World"}


#@app.get("/ids/")
#def get_ids():

#    Liste_id = data["SK_ID_CURR"].to_list()
#    id = [int(c) for c in Liste_id]
#    return id

@app.get("/ids/")
def get_ids():

    id = data["SK_ID_CURR"]
    return id


@app.get("/data_client/{client_id}")
def identite_client(client_id):

    try:
        client_id = int(client_id)
    except:
        raise AttributeError(f"Problem with client_id : {client_id}, {type(client_id)}")

    data_client = data.loc[data["SK_ID_CURR"] == client_id]
    return data_client


@app.get("/credit_moyen/")
def credit_moy():

    credits_moy = round(data["AMT_CREDIT"].mean(), 2)
    return credits_moy

@app.get("/test_nb_colonne/")
def nb_colonne():

    nb_colonne =X_train_sample.shape[1]
    return nb_colonne

@app.get("/nb_credit/")
def nb_credits():
    """# Informations sur dataset"""

    nb_credits = data.shape[0]

    return nb_credits


@app.get("/rev_moyen/")
def rev_moy():
    """# Informations sur dataset"""

    rev_moy = round(data["AMT_INCOME_TOTAL"].mean(), 2)

    return rev_moy


@app.get("/age/")
def load_age_population():
    """# Informations sur dataset"""

    data_age = round(-(data["DAYS_BIRTH"] / 365), 2)

    return data_age


@app.get("/data_income/")
def load_income_population():

    df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income["AMT_INCOME_TOTAL"] < 300000, :]

    return df_income

@app.get("/data_age/")
def load_income_population():

    df_age = pd.DataFrame(data["DAYS_BIRTH"])

    return df_age

########################################""
# juste renvoie le describe...
########################################"


# Calcul prédiction
@app.get("/prediction/{client_id}")
def load_prediction(client_id):

    try:
        client_id = int(client_id)
    except:
        raise AttributeError(f"Problem with client_id : {client_id}, {type(client_id)}")

        
    id = data["SK_ID_CURR"].to_list()
    nbligne=id.index(client_id)    
        
    prediction = proba_df.iat[nbligne, 1]

    if proba_df.iat[nbligne, 1] * 100 > 50:
        statut = 0
    else:
        statut = 1

    return statut


# Shap value
#@app.get("/shap/{client_id}")
#def shap_value(client_id):

#    try:
#        client_id = int(client_id)
#    except:
#        raise AttributeError(f"Problem with client_id : {client_id}, {type(client_id)}")
#        
#    id = data["SK_ID_CURR"].to_list()
#    nbligne=id.index(client_id) 
#
 #   shap_id = shap_vals[nbligne][:, 0].values
#    shap_id_dict = dict(enumerate(shap_id.flatten(), 1))

#    return shap_id_dict

@app.get("/shap/{client_id}")
def shap_values(client_id):
    """
    Get the shap values for a selected customer.
    These features are have the most impact on model prediction for this specific customer (local explainer)

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    shap._explanation.Explanation
        Shap explanation object for a specific customer

    """
    id = data["SK_ID_CURR"]
    try:
        client_id = int(client_id)
    except:
        raise AttributeError(f"Problem with client_id : {client_id}, {type(client_id)}")

    idx = id.index.get_loc(client_id)
    return explanation_to_dict(shap_vals[idx])
