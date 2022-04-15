# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 01:23:10 2022

@author: guial
"""

import streamlit as st 
import numpy as np 
import pandas as pd
import shap # v0.39.0
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import plotly.graph_objects as go
import requests
import json




st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")

st.title("Prêt à Dépenser")

st.write("""
# Dashboard Interactif pour évaluez la solvabilité d'un client
""")

test_X = pd.read_csv('test_X.csv')
test_X = test_X.drop(["Unnamed: 0"], axis=1)
# test_X = test_X.loc[range(1000),]

SK_ID_CURR_test_X = pd.read_csv('SK_ID_CURR_test_X.csv')
SK_ID_CURR_test_X = SK_ID_CURR_test_X.drop(["Unnamed: 0"], axis=1)
# SK_ID_CURR_test_X = SK_ID_CURR_test_X.loc[range(1000),]

test_X_2 = pd.read_csv('test_X_2.csv')
test_X_2 = test_X_2.drop(["Unnamed: 0"], axis=1)
# test_X_2 = test_X_2.loc[range(1000),]

app_train = pd.read_csv('app_train.csv')
app_train = app_train.drop(["Unnamed: 0"], axis=1)

image = Image.open('Pret_A_Depenser.png')

pickle_in = open("xgb_cl_undersampling.pkl","rb")
xgb_cl_undersampling = pickle.load(pickle_in)

client_number = st.selectbox("Séléctionnez un client", (i for i in SK_ID_CURR_test_X['SK_ID_CURR']))

data_df = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),:]
    
# Create prediction
prediction = float(xgb_cl_undersampling.predict_proba(data_df)[0][1])



# Largeur de la barre latérale
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
st.sidebar.image(image, width=240, caption=" Dashboard - Aide à la décision",
                  use_column_width='always')


# Score du client en pourcentage arrondi et nombre entier
score_client = int(np.rint(prediction * 100))


# Graphique de jauge du cédit score ==========================================
fig_jauge = go.Figure(go.Indicator(
    mode = 'gauge+number',
    # Score du client en % df_dashboard['SCORE_CLIENT_%']
    value = score_client,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'Crédit score du client (en %)', 'font': {'size': 24}},
    
    gauge = {'axis': {'range': [None, 100],
                      'tickwidth': 3,
                      'tickcolor': 'darkblue'},
              'bar': {'color': 'white', 'thickness' : 0.25},
              'bgcolor': 'white',
              'borderwidth': 2,
              'bordercolor': 'gray',
              'steps': [{'range': [0, 44.99], 'color': 'Green'},
                        {'range': [45, 55], 'color': 'Orange'},
                        {'range': [55, 100], 'color': 'Red'}],
              'threshold': {'line': {'color': 'white', 'width': 10},
                            'thickness': 0.8,
                            'value': score_client}}))

fig_jauge.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': 'darkblue', 'family': 'Arial'},
                        margin=dict(l=0, r=0, b=0, t=0, pad=0))

with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig_jauge)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Texte d'accompagnement de la jauge
        if 0 <= score_client < 25:
            score_text = 'Crédit score : EXCELLENT'
            st.success(score_text)
        elif 25 <= score_client < 45:
            score_text = 'Crédit score : BON'
            st.success(score_text)
        elif 45 <= score_client < 55:
            score_text = 'Crédit score : MOYEN'
            st.warning(score_text)
        else :
            score_text = 'Crédit score : BAS'
            st.error(score_text)
        st.write("")    
        


# # 2.2 Affichage de ses informations générales dans la barre latérale

app_train['AGE'] = (app_train['DAYS_BIRTH']/-365).astype(int)


test_X_2['AGE'] = (test_X_2['DAYS_BIRTH']/365).astype(int)

test_X_2['TYPE_DE_CONTRAT'] = test_X_2["NAME_CONTRACT_TYPE"].map({0: 'Cash loans', 1: 'Revolving loans'})

test_X_2['CODE_GENDER'] = test_X_2["CODE_GENDER_M"].map({0: 'Women', 1: 'Men'})

test_X_2['NAME_HOUSING_TYPE'] = test_X_2["NAME_HOUSING_TYPE_Co-op apartment"].map({0: '', 1: 'Co-op apartment'})+test_X_2["NAME_HOUSING_TYPE_House / apartment"].map({0: '', 1: 'House / apartment'})+test_X_2["NAME_HOUSING_TYPE_Municipal apartment"].map({0: '', 1: 'Municipal apartment'})+test_X_2["NAME_HOUSING_TYPE_Office apartment"].map({0: '', 1: 'Office apartment'})+test_X_2["NAME_HOUSING_TYPE_Rented apartment"].map({0: '', 1: 'Rented apartment'})+test_X_2["NAME_HOUSING_TYPE_With parents"].map({0: '', 1: 'With parents'})

test_X_2['NAME_FAMILY_STATUS'] = test_X_2["NAME_FAMILY_STATUS_Civil marriage"].map({0: '', 1: 'Civil marriage'})+test_X_2["NAME_FAMILY_STATUS_Married"].map({0: '', 1: 'Married'})+test_X_2["NAME_FAMILY_STATUS_Separated"].map({0: '', 1: 'Separated'})+test_X_2["NAME_FAMILY_STATUS_Single / not married"].map({0: '', 1: 'Single / not married'})+test_X_2["NAME_FAMILY_STATUS_Unknown"].map({0: '', 1: 'Unknown'})+test_X_2["NAME_FAMILY_STATUS_Widow"].map({0: '', 1: 'Widow'})

test_X_2['NAME_EDUCATION_TYPE'] = test_X_2["NAME_EDUCATION_TYPE_Academic degree"].map({0: '', 1: 'Academic degree'})+test_X_2["NAME_EDUCATION_TYPE_Higher education"].map({0: '', 1: 'Higher education'})+test_X_2["NAME_EDUCATION_TYPE_Incomplete higher"].map({0: '', 1: 'Incomplete higher'})+test_X_2["NAME_EDUCATION_TYPE_Lower secondary"].map({0: '', 1: 'Lower secondary'})+test_X_2["NAME_EDUCATION_TYPE_Secondary / secondary special"].map({0: '', 1: 'Secondary / secondary special'})


tableau = test_X_2[['TYPE_DE_CONTRAT', 'CODE_GENDER',
                    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY','NAME_HOUSING_TYPE',
                    'NAME_FAMILY_STATUS','NAME_EDUCATION_TYPE','AGE']]


st.sidebar.write("#### Type de prêt")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'TYPE_DE_CONTRAT'].values[0])
st.sidebar.write("#### Sexe")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'CODE_GENDER'].values[0])
st.sidebar.write("#### Revenu total (AMT_INCOME_TOTAL)")
st.sidebar.write(int(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'AMT_INCOME_TOTAL'].values[0]))
st.sidebar.write("#### Montant du crédit (AMT_CREDIT)")
st.sidebar.write(int(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'AMT_CREDIT'].values[0]))
st.sidebar.write("#### Montant des annuités (AMT_ANNUITY)")
st.sidebar.write(int(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'AMT_ANNUITY'].values[0]))
st.sidebar.write("#### Situation familiale")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'NAME_FAMILY_STATUS'].values[0])
st.sidebar.write("#### Niveau d'éducation")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'NAME_EDUCATION_TYPE'].values[0])
st.sidebar.write("#### Type de logement")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'NAME_HOUSING_TYPE'].values[0])
st.sidebar.write("#### Age (AGE)")
st.sidebar.write(tableau.loc[tableau.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),'AGE'].values[0])



with st.container():
    col1, col2 = st.columns([1.6, 1.5])
    variable = st.selectbox("Séléctionnez une variable",(i for i in app_train[['AMT_INCOME_TOTAL', 'AMT_CREDIT',
                                                                                'AMT_ANNUITY','AGE','EXT_SOURCE_1',
                                                                                'EXT_SOURCE_2','EXT_SOURCE_3']]))
    
    with col1:
        explainer = shap.TreeExplainer(xgb_cl_undersampling, model_output='probability', feature_perturbation = 'interventional', data=test_X)
        shap_values = explainer(test_X)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index)])
        st.pyplot(fig)
        
    
    with col2:
        fig = plt.figure()
        plt.style.use('fivethirtyeight')
        # plot repaid loans
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, variable], label = 'Remboursement')
        # plot loans that were not repaid
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, variable], label = 'Défaut')
        # Label the plots
        plt.axvline(x = test_X_2.loc[test_X_2.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==client_number].index),variable].values[0], color = 'r', linestyle = '-')
        plt.title('%s' % variable)
        plt.xlabel('valeur'); plt.ylabel('Densitée');
        plt.legend()
        st.pyplot(fig)
        
                        
# streamlit run Streamlit_2.py
