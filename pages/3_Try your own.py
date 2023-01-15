import streamlit as st
import pandas as pd
import numpy as np
import random as rnd
import pickle  
import time

model = pickle.load(open('models/gbm_20_cr_all.pkl', 'rb'))

@st.cache
def testDfCreate(lst):
    df = pd.DataFrame()
    for i in range(1, 25):
        df['Q' + str(i)] = 1
    df.loc[len(df)] = lst
    return df

def predictDf(df):
        prediction = model.predict(df)
        if prediction[0] == 0 :
            st.success("A regular responder!")
            st.balloons()
        else:
            st.warning("Carelessness detected!")

st.set_page_config(
    page_title="Try Your Own",
    page_icon="",
)

st.write("Take a survey and test if it is considered careless")

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