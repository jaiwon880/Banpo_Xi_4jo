import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from data import get_city_list, get_gu_list, get_town_list, get_village_list
from service import get_filtered_data, handle_preprocessing, read_data
import datetime
import pandas as pd
import numpy as np
import joblib
from ML_Model import *
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt

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
    area = 0
    year_apt = 0
    genre = 0
    col1,col2 = st.columns([1, 1])
    with col1 :
        area = st.slider('전용 면적을 선택해 주세요', 0.0, 300.0)

        options = {
            "중개거래": 0,
            "직거래": 1,
        }
        genre = st.radio("거래 유형을 선택해 주세요", list(options.keys()))
        genre = options[genre]
        st.markdown(f"<div style='margin-top: 25px; margin-right: 20px;'></div>", unsafe_allow_html=True)
    with col2 :
        year_apt = st.slider('건축 년도를 선택해 주세요', min_value = 1940, max_value=2023,step=1)
        st.markdown(f"<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button('현재 금리 적용'):
            today = datetime.date.today()
            st.write(f'현재 선택한 금리는 {3.75} 입니다')
        else:
            today = datetime.date.today()

    if st.button('입력 완료',key='set_data'):
        st.write("입력이 완료 되었습니당")            
        input_data = pd.DataFrame(np.array([area,year_apt,genre,3.75]).reshape(1,-1),
                                    columns = ['전용면적(㎡)','건축년도','거래유형','금리'])
        return input_data

    
def contents():
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['df',"Linear Regressor", 'KNN', "Decision Tree", 'Random Forest', "XGBoost", "LightGBM"])
    data = read_data()
    # try:
    with tab0:
        background()
        aa=col_()
        st.write(aa)

    with tab1: 
        tab1.subheader("📈Linear Regression📈")
        lr_model = lr()
        Linear_(lr_model,aa)
    with tab2: 
        tab2.subheader("🤝KNN🤝")
        knn_model = knn()
        KNN_(knn_model,aa)
    with tab3:
        tab3.subheader("🌲Decision Tree🌲")
        decision = dct()
        DCT_(decision,aa)
    with tab4:
        tab4.subheader("🌳Random Forest🌳") 
        rf = rdf()
        RDF_(rf,aa)
    with tab5:
        tab5.subheader("💪XGBoost💪") 
        xgb_model = xgb()
        XGB_(xgb_model,aa)
    with tab6: 
        tab6.subheader("⚡️LightGBM⚡️")
        lgbmR = lgbm()
        LGBM_(lgbmR,aa)
    # except:
    #     pass

def background():
    st.dataframe(handle_preprocessing())

def load_data():
    datas = handle_preprocessing()
    train = datas.loc[datas.index < '2023-01-01']
    test = datas.loc[datas.index >= '2023-01-01']
    X_train = train.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_train = train['평당가']
    X_test = test.drop(['시군구','거래금액(만원)','평당가'],axis=1)
    y_test = test['평당가']

    return X_train,y_train,X_test,y_test

# lr 모델
def lr():
    X_train,y_train,X_test,y_test = load_data()
    models = []
    model = LinearRegression()
    model.fit(X_train,y_train)

    pred= model.predict(X_test)
    rmse = mean_squared_error(y_test,pred)**0.5
    st.write('모델의 RMSE 값',rmse)
    st.write('모델의 예측 값',pred)

    return model


# knn 모델
def knn():
    X_train, y_train, X_test, y_test = load_data()
    r2_scores=[]
    rmse_ = []

    for i in range(1,6):
        model = KNeighborsRegressor(i,weights='distance')
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        rmse = mean_squared_error(y_test,pred)**0.5
        r2=r2_score(y_test,pred)
        r2_scores.append(r2)
        rmse_.append(rmse)

    st.write(r2_scores)
    st.write(rmse_)

    return model

# 랜덤포레스트 모델
def rdf():
    X_train,y_train,X_test,y_test = load_data()
    rmse_ = []
    models = []
    min_estimators = 50
    max_estimators = 200
    step = 50
    n_estimators_range = range(min_estimators, max_estimators+1, step)
    r2_scores = []
    for n_estimators in n_estimators_range:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test,y_pred)**0.5
        rmse_.append(rmse)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='실제값'))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='예측값'))
    fig.update_layout(title='XGBoost 모델 예측 결과',
                    xaxis_title='데이터 인덱스',
                    yaxis_title='예측값')
    st.plotly_chart(fig)
    return model

# 결정트리 모델
def dct():
    X_train,y_train,X_test,y_test = load_data()

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

    return model

# XGBoost 모델
def xgb():
    X_train, y_train, X_test, y_test = load_data()
    # 모델 훈련 및 예측
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, min_child_weight=3, colsample_bytree=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # 모델 성능 평가
    rmse = mean_squared_error(y_test, pred)**0.5
    # 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='실제 값'))
    fig.add_trace(go.Scatter(x=list(range(len(pred))), y=pred, mode='lines', name='모델 예측 값'))
    fig.update_layout(title='XGBoost 모델 예측 결과',
                    xaxis_title='데이터 인덱스',
                    yaxis_title='예측값')
    st.plotly_chart(fig)
    st.write('모델의 RMSE:', rmse)
    
    return model




# LGBM 모델
def lgbm():
    X_train,y_train,X_test,y_test = load_data()

    models = []
    for i in range(0,5):
        if i==0:
            continue
        model = LGBMRegressor()
        model.fit(X_train,y_train)

        pred=model.predict(X_test)
        rmse = mean_squared_error(y_test,pred)**0.5
        # 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='실제 값'))
    fig.add_trace(go.Scatter(x=list(range(len(pred))), y=pred, mode='lines', name='모델 예측 값'))
    fig.update_layout(title='LGBM 모델 예측 결과',
                    xaxis_title='데이터 인덱스',
                    yaxis_title='예측값')
    st.plotly_chart(fig)
    st.write('모델의 RMSE:', rmse)
    
    return model


if __name__ == '__main__':
    main()