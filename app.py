# ====================== IMPORT PACKAGES ==============

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 

import streamlit as st
import base64
import seaborn as sns
import pickle

# ------------ TITLE 

st.markdown(f'<h1 style="color:#964B00;text-align: center;font-size:38px;font-family:Caveat, sans-serif">{"A Hybrid Sampling Approach for Credit Card Fraud Detection Using ML"}</h1>', unsafe_allow_html=True)


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')   




# ================== PREDICTION  ====================
st.write("---------------------------------------------")

st.markdown(f'<h1 style="color:#bd142b;text-align: center;font-size:28px;font-family:Caveat, sans-serif">{"Prediction"}</h1>', unsafe_allow_html=True)
st.write("---------------------------------------------")

with open('rf.pickle', 'rb') as f:
    rf = pickle.load(f)


# with open('iso_forest.pickle', 'rb') as f:
#     isolation = pickle.load(f)


# --- PREDICTION 


# Collect user inputs for all features
time = st.number_input('Enter Number of seconds elapsed between this transaction and the first transaction ', min_value=0, step=1)
v1 = st.number_input('V1')
v2 = st.number_input('V2')
v3 = st.number_input('V3')
v4 = st.number_input('V4')
v5 = st.number_input('V5')
v6 = st.number_input('V6')
v7 = st.number_input('V7')
v8 = st.number_input('V8')
v9 = st.number_input('V9')
v10 = st.number_input('V10')
v11 = st.number_input('V11')
v12 = st.number_input('V12')
v13 = st.number_input('V13')
v14 = st.number_input('V14')
v15 = st.number_input('V15')
v16 = st.number_input('V16')
v17 = st.number_input('V17')
v18 = st.number_input('V18')
v19 = st.number_input('V19')
v20 = st.number_input('V20')
v21 = st.number_input('V21')
v22 = st.number_input('V22')
v23 = st.number_input('V23')
v24 = st.number_input('V24')
v25 = st.number_input('V25')
v26 = st.number_input('V26')
v27 = st.number_input('V27')
v28 = st.number_input('V28')
amount = st.number_input('Amount')


aa = st.button("PREDICT")

if aa:
    
    user_input = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                            v21, v22, v23, v24, v25, v26, v27, v28, amount]]).reshape(1, -1)
                            

    
    # Data = np.array([a1,a2,a3,a4,a5,a6,int(a7),int(a8),int(a9),int(a10),a11])
    # st.write(Data)
    
    # -- fraud

     
    pred_rf = rf.predict(user_input)

    pred_rf = int(pred_rf)
    
    if pred_rf == 0:
        
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified - NON FRAUD"}</h1>', unsafe_allow_html=True)
        st.write("----------------------------------------------------------------------------------------------")


    elif pred_rf == 1:
        
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified  - FRAUD"}</h1>', unsafe_allow_html=True)
        st.write("----------------------------------------------------------------------------------------------")



  





















