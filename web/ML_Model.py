import streamlit as st

def Linear_(model,data):
    pred = model.predict(data)
    st.write(pred)

# knn 모델
def KNN_(model,data):
    pred = model.predict(data)
    st.write(pred)

# 랜덤포레스트 모델
def RDF_(model,data):
    pred = model.predict(data)
    st.write(pred)

# 결정트리 모델
def DCT_(model,data):
    pred = model.predict(data)
    st.write(pred)

# XGBoost 모델
def XGB_(model,data):
    pred = model.predict(data)
    st.write(pred)
# LGBM 모델
def LGBM_(model,data):
    pred = model.predict(data)
    st.write(pred)