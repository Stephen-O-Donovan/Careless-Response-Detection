import streamlit as st
import pandas as pd
import numpy as np
import random as rnd
import pickle  

@st.cache
def testDfCreate(lst):
    df = pd.DataFrame()
    for i in range(1, 25):
        df['Q' + str(i)] = 1
    df.loc[len(df)] = lst
    return df

def predictDf(df, user_selection, scoring=False):
    file = 'models/'+ user_selection +'.pkl'
    model = pickle.load(open(file, 'rb'))
    prediction = model.predict(df)
    if scoring:
        return prediction[0]
    if prediction[0] == 0 :
        st.success("A regular responder!")
        st.balloons()
    else:
        st.warning("Carelessness detected!")

@st.cache
def setType(model_select):

    switch={
      'Gradient Boosted ':'gbm',
      'Random Forest':'rf',
      'K-Nearest Neighbours':'knn',
      'Support Vector Machines':'svm',
      'Neural Net':'nnet'
      }
    return switch.get(model_select,"gbm")


st.set_page_config(
    page_title="Select Models",
    page_icon="",
)

st.write("Select from different models and settings")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.write('Select what kind of responders will be detected')
    cr_type = st.selectbox('Survey Type',options=['all', 'human', 'computer'])

with col2:
    st.write('Select the expected percentage rate of carelessness')
    cr_rate = str(st.selectbox('Careless Rate',options=[5, 10, 15, 20]))

with col3:
    st.write('Select the model type to use for detection')
    cr_model = setType(st.selectbox('Model', 
            options=['Random Forest', 'Gradient Boosted', 'K-Nearest Neighbours', 
            'Support Vector Machines', 'Neural Net']))

user_selection = cr_model + '_' + cr_rate + '_cr_' + cr_type

if st.button("Predict based on regular responder"):
    st.write('Running ' + user_selection)
    rr_list = [4,4,3,4,5,5,4,5,5,5,4,4,3,4,4,4,4,5,5,4,4,4,5,5]
    df = testDfCreate(rr_list)
    
    predictDf(df, user_selection)

if st.button("Predict based on random generated surveys"):
    st.write('Running ' + user_selection)
    lst = []
    for a in range(1, 25):
        lst.append(rnd.randint(1,5))

    df = testDfCreate(lst)
    predictDf(df, user_selection)

if st.button("Score based on 100 random generated surveys"):
    st.write('Running ' + user_selection)
    score = 0
    my_bar = st.progress(0)
    for run in range(100):
        my_bar.progress(run + 1)
        lst = []
        for a in range(1, 25):
            lst.append(rnd.randint(1,5))

        df = testDfCreate(lst)
        score += predictDf(df, user_selection, True)
    st.write(score)