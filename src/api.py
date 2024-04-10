"""
routes file
"""

from fastapi import FastAPI, Response  # , UploadFile

from .preprocess import *


app = FastAPI()


@app.get("/")
def read_main():

    return {"msg": "Hello World"}


@app.get("/ids/")
def get_ids():

    id = data["SK_ID_CURR"].to_dict()
    return id


@app.get("/data_client/")
def identite_client(id):

    data_client = data[data["SK_ID_CURR"] == id]
    return data_client


@app.get("/credit_moyen/")
def credit_moy():

    credits_moy = round(data["AMT_CREDIT"].mean(), 2)
    return credits_moy


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
def load_age_population(data):
    """# Informations sur dataset"""

    data_age = round(-(data["DAYS_BIRTH"] / 365), 2)

    return data_age


@app.get("/data_plot/")
def load_income_population(sample):

    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income["AMT_INCOME_TOTAL"] < 200000, :]

    return df_income


# Calcul prédiction
@app.get("/prediction/")
def load_prediction(id):

    data_ID = data[["SK_ID_CURR"]]
    y_pred_lgbm_proba = model_LGBM.predict_proba(X_train_sample)
    y_pred_lgbm_proba_df = pd.DataFrame(
        y_pred_lgbm_proba, columns=["proba_classe_0", "proba_classe_1"]
    )
    y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df, data_ID], axis=1)

    y_pred_lgbm_proba_df = y_pred_lgbm_proba_df[
        y_pred_lgbm_proba_df["SK_ID_CURR"] == id
    ]
    prediction = y_pred_lgbm_proba_df.iat[0, 1]

    if y_pred_lgbm_proba_df.iat[0, 1] * 100 > 50:
        statut = "Client risqué"
    else:
        statut = "Client non risqué"

    return prediction, statut


# Shap value
@app.get("/shap/")
def shap_value(id):

    nbligne = data.loc[data["SK_ID_CURR"] == int(id)].index.item()
    # fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.Explainer(model_LGBM)
    shap_values = explainer.shap_values(X_train_sample)
    shap_vals = explainer(X_train_sample)
    shap_id = shap_vals[nbligne][:, 0].values
    # st.pyplot(fig)

    return shap_id
