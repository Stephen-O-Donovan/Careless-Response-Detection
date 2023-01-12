import streamlit as st
import pandas as pd
import numpy as np
import random as rnd
import pickle  
import time

st.write("Careless Response Detection")

@st.cache
def testDfCreate(lst):
    df = pd.DataFrame()
    for i in range(1, 25):
        df['Q' + str(i)] = 1
    df.loc[len(df)] = lst
    return df

def predictDf(df):
        prediction = rf_model.predict(df)
        if prediction[0] == 0 :
            st.success("A regular responder!")
            st.balloons()
        else:
            st.warning("Carelessness detected!")

app_mode = st.sidebar.selectbox('Select Page',['Home', 'Random Forest','Try Your Own']) 
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
        rr_list = [4,4,3,4,5,5,4,5,5,5,4,4,3,4,4,4,4,5,5,4,4,4,5,5]
        df = testDfCreate(rr_list)
        predictDf(df)

    if st.button("Predict based on random generated"):

        lst = []
        for a in range(1, 25):
            lst.append(rnd.randint(1,5))

        df = testDfCreate(lst)
        predictDf(df)

elif app_mode=="Try Your Own":
    rf_model = pickle.load(open('models/rf_20_all.pkl', 'rb'))
    lst = []
    placeholder = st.empty()
    with placeholder.container():
        for i in range(1,25):
            lst.append(st.selectbox('Q'+str(i),options=[1,2,3,4,5], index=0, key=i+int(time.time()) ))
    if st.button('Randomise Answers'):
        lst = []
        placeholder.empty()
        with st.spinner('Randomising...'):
            time.sleep(1)
        with placeholder.container():
            for i in range(1,25):
                lst.append(st.empty().selectbox('Q'+str(i),options=[1,2,3,4,5], index=rnd.randint(0,4), key=i+25+int(time.time())))
            # st.write(lst[i]) #.index=rnd.randint(0,4)
    if st.button('Test Survey'):
        df = testDfCreate(lst)
        predictDf(df)
