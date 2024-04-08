#####################################
########### Script pour API #########
#####################################
import uvicorn

from fastapi import FastAPI, Response #, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import re
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
api = FastAPI()

# 3. Imports of required pkl files:

pkl_1= open("pickle/best_lgbm.pkl","rb")
best_LGBM=pd.read_pickle(pkl_1)

pkl_2= open("pickle/data.pkl","rb")
data=pd.read_pickle(pkl_2)

pkl_3 = open("pickle/encodage.pkl","rb")
encodage=pd.read_pickle(pkl_3)

pkl_4 = open("pickle/imputer.pkl","rb")
imputer=pd.read_pickle(pkl_4)

pkl_5 = open("pickle/scale.pkl","rb")
scale=pd.read_pickle(pkl_5)

pkl_6 = open("pickle/sampler.pkl","rb")
sampler=pd.read_pickle(pkl_6)

pkl_7 = open("pickle/estimateur.pkl","rb")
estimateur=pd.read_pickle(pkl_7)

pkl_8= open("pickle/trans_Imput_Scale.pkl","rb")
trans_imput_scale=pd.read_pickle(pkl_8)

y=data['TARGET']
X=data.drop(['SK_ID_CURR','TARGET'],axis=1)




cat_feat=X.select_dtypes(include=object).columns.to_list()
num_feat=X.select_dtypes(exclude=object).columns.to_list()

X_cat=X[cat_feat]
X_num=X[num_feat]

encoder = encodage
encoder.fit(X_cat)

feature_names_out = encoder.get_feature_names_out(input_features=X_cat.columns) # Obtenir les noms de colonnes étendus
X_cat_encoded = encoder.transform(X_cat) # Transformer les données
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=feature_names_out, index=X.index)# Convertir la matrice creuse en DataFrame
X_VF = pd.concat([X_num,X_cat_encoded_df],axis=1)
X_VF_Imputer=pd.DataFrame(imputer.fit_transform(X_VF),columns=X_VF.columns)
data_trans=pd.DataFrame(scale.fit_transform(X_VF_Imputer),columns=X_VF_Imputer.columns)

X_train,X_test,y_train,y_test=train_test_split(data_trans,y,test_size=0.2)

X_train_sample,y_train_sample=sampler.fit_resample(X_train,y_train)

X_train_sample = X_train_sample.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

model_LGBM = estimateur.fit(X_train_sample, y_train_sample)
shap_values_lgbm =  model_LGBM.predict(X_train_sample, pred_contrib=True)




#shap-value
# explain the model
explainer = shap.TreeExplainer(model_LGBM)
shap_vals = explainer(X_train_sample)



@api.get('/')
def index():
    return {'message': 'Hello. API en cours'} 

@api.get("/") 
def get_ids() -> dict:
    id=data['SK_ID_CURR'].to_dict()
    return id
    
# Identité client
@api.get("/") 
def identite_client(id):
        data_client = data[data.index == int(id)]
        return data_client
    
# Informations sur dataset
@api.get('/')
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]
    return nb_credits, rev_moy, credits_moy



    
# Informations sur dataset
@api.get("/") 
def load_age_population(data):
    data_age = round(-(data["DAYS_BIRTH"]/365), 2)
    return data_age
    
   
   
@api.get("/") 
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income
    
# Calcul prédiction    
@api.get("/")
def load_prediction(id):
    data_ID=data[['SK_ID_CURR']]
    y_pred_lgbm_proba = model_LGBM.predict_proba(X_train_sample)
    y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
    y_pred_lgbm_proba_df=pd.concat([y_pred_lgbm_proba_df, data_ID], axis=1)
        
    y_pred_lgbm_proba_df=y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['SK_ID_CURR']==int(id)]
    prediction=y_pred_lgbm_proba_df.iat[0,1]
        
    if y_pred_lgbm_proba_df.iat[0,1]*100>50 : 
            statut="Client risqué" 
    else :
            statut="Client non risqué"
    return prediction,statut
    

# Calcul prédiction    
@api.get("/")
def shap_value(id):
    nbligne=data.loc[data['SK_ID_CURR'] == int(id)].index.item()
    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.Explainer(model_LGBM)
    shap_values = explainer.shap_values(X_train_sample)
    shap_vals = explainer(X_train_sample)
    shap_id=shap_vals[nbligne][:, 0].values
    #st.pyplot(fig)
    return shap_id

    

if __name__ == '__main__':
    #uvicorn.run(api, host='127.0.0.1', port=8000)
    # version KO du 18/02/24: uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run(api, host='127.0.0.1', port=8000)


