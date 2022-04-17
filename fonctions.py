import pickle
import pandas as pd
import numpy as np

pickle_in = open("xgb_cl_undersampling.pkl","rb")
xgb_cl_undersampling = pickle.load(pickle_in)

test_X = pd.read_csv('test_X.csv')
test_X = test_X.drop(["Unnamed: 0"], axis=1)

SK_ID_CURR_test_X = pd.read_csv('SK_ID_CURR_test_X.csv')
SK_ID_CURR_test_X = SK_ID_CURR_test_X.drop(["Unnamed: 0"], axis=1)



    
def API_prediction(lien,numero_client):
    data_df = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==numero_client].index),:]
    resultat = float(xgb_cl_undersampling.predict_proba(data_df)[0][1])
    return resultat