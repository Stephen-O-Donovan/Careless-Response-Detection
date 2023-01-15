
import streamlit as st
import pickle  
import pandas as pd

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