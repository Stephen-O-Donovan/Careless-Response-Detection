import streamlit as st
import pandas as pd
import numpy as np
import random as rnd
import pickle  

st.write("Careless Response Detection")

app_mode = st.sidebar.selectbox('Select Page',['Home', 'Random Forest']) 
if app_mode=='Home':
    st.title('Home Page :')  
    st.write('Select a model from the sidebar')
    st.markdown('''
    This site is based on the major thesis of my Data Science Masters Degree.
    The aim of the project was to test the effectiveness of various machine learning models
    in detecting careless responders in survey data.
    ''')

elif app_mode=="Random Forest":
    rf_model = pickle.load(open('models/rf_20_all.pkl', 'rb'))

    if st.button("Predict based regular responder"):
        df = pd.DataFrame()
        for i in range(1, 25):
            df['Q' + str(i)] = 1
        rr_list = [4,4,3,4,5,5,4,5,5,5,4,4,3,4,4,4,4,5,5,4,4,4,5,5]
        df.loc[len(df)] = rr_list

        prediction = rf_model.predict(df)
        if prediction[0] == 0 :
            st.success("A regular responder!")
            st.balloons()
        else:
            st.warning("Carelessness detected!")

    if st.button("Predict based on random generated"):

        
        df = pd.DataFrame()
        for i in range(1, 25):
            df['Q' + str(i)] = 1
        lst = []
        for a in range(1, 25):
            lst.append(rnd.randint(1,5))

        df.loc[len(df)] = lst

        prediction = rf_model.predict(df)
        if prediction[0] == 0 :
            st.success("A regular responder!")
            st.balloons()
        else:
            st.warning("Carelessness detected!")