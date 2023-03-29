import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from data import get_city_list, get_gu_list, get_town_list, get_village_list
from service import get_filtered_data, handle_preprocessing
import datetime
import pandas as pd
import numpy as np
import joblib


def main():
    with st.sidebar: sidebar()
    contents()


def sidebar() :
    title =  '지역을 선택해주세요.'
    st.title(title)
    label_city_list = '시/도 선택'
    label_gu_list = '시/군/구 선택'
    label_town_list = '읍/면/동 선택'
    city_choice = st.selectbox(label_city_list, get_city_list(),
    key='city')
    gu_choice = st.selectbox(label_gu_list, get_gu_list(city_choice),
    key='gu')
    town_choice = st.selectbox(label_town_list, get_town_list(city_choice, gu_choice),
    key='town')
    vl = get_village_list(city_choice, gu_choice, town_choice)
    if len(vl)>0:
        village_choice = st.selectbox(label_town_list, vl, key='village')

    else:
        st.session_state['village'] = ''


def col_():
        col1,col2 = st.columns([1, 1])
        with col1 :
            st.slider('전용 면적을 선택해 주세요', 0.0, 300.0)
            genre = st.radio(
                "거래 유형을 선택해 주세요",
                ('중개거래', '직거래'))
        with col2 :
            st.slider('건축 년도를 선택해 주세요', min_value = 1940, max_value=2023, step=1)
            if st.button('현재 금리 적용'):
                today = datetime.date.today()
            else:
                today = datetime.date.today()
        if st.button('예측'):
            st.write("아파트 실거래가 예측 값 입니다")
        else:
            st.write("")

        

def background():
    st.dataframe(handle_preprocessing())


def load_data():
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

# lr 모델
def lr():
    X_train,y_train, X_test, y_test = load_data()

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train,y_train)

        pred= model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        
        models.append(rmse)
    st.write('모델의 RMSE 값',models)
    st.write('모델의 예측 값',pred)


# knn 모델
def knn():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = KNeighborsRegressor(n_neighbors=i,weights='distance')
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        
        models.append(rmse)

    st.write(models)
    st.write('모델의 예측 값',pred)

# 랜덤포레스트 모델
def rdf():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = RandomForestRegressor(n_estimators=150,max_depth=4)
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        
        models.append(rmse)

    st.write(models)
    st.write('모델의 예측 값',pred)

# 결정트리 모델
def dct():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = DecisionTreeRegressor(random_state=i)
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        
        models.append(rmse)

    st.write(models)
    st.write('모델의 예측 값',pred)

# XGBoost 모델
def xgb():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, min_child_weight=3, colsample_bytree=1)
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        models.append(rmse)

    st.write(models)
    st.write('모델의 예측 값',pred)

# LGBM 모델
def lgbm():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = LGBMRegressor(num_leaves=16, max_depth=4, learning_rate=0.1)
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        models.append(rmse)

    st.write(models)
    st.write('모델의 예측 값',pred)


col_()

def contents():
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['DataFrame', "Linear Regressor", 'KNN', "Decision Tree", 'Random Forest', "XGBoost", "LightGBM"])
    with tab0:
        background()
    with tab1: 
        tab1.subheader("Linear Regression")
        lr() 
    with tab2: 
        tab2.subheader("KNN")
        knn()
    with tab3:
        tab3.subheader("Decision Tree")
        dct()
    with tab4:
        tab4.subheader("Random Forest") 
        rdf()
    with tab5:
        tab5.subheader("XGBoost") 
        xgb()
    with tab6: 
        tab6.subheader("LightGBM")
        lgbm()



if __name__ == '__main__':
    main()