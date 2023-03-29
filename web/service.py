import streamlit as st
from data import read_data, check_contain
import pandas as pd

def get_filtered_data():
    df = read_data()
    ss = st.session_state
    city_choice, gu_choice, town_choice, village_choice = [
        ss[k] for k in ['city', 'gu', 'town', 'village']
    ]
    return df[
        check_contain(df['시군구'], city_choice)
        & check_contain(df['시군구'], gu_choice)
        & check_contain(df['시군구'], town_choice)
        & check_contain(df['시군구'], village_choice)
    ]

def handle_preprocessing():
    datas = get_filtered_data()
    datas['거래금액(만원)']=(datas['거래금액(만원)'].str.replace(',','')).str.strip().astype(int)
    datas["평당가"] = (datas["거래금액(만원)"] / datas["전용면적(㎡)"]*0.3025) # 평당가 계산하여 새로운 컬럼 추가
    aa=datas['계약년월'].astype(str)
    bb=datas['계약일'].astype(str)
    datas['date'] = aa+bb
    datas['date'] = pd.to_datetime(datas['date'], format='%Y%m%d')
    datas.drop(['계약년월','계약일'],axis=1,inplace=True)
    datas.sort_values('date',ascending = True, inplace=True)
    datas.set_index('date',inplace=True)
    datas['거래유형'] = datas['거래유형'].apply(lambda x:0 if x=='중개거래' else 1)
    datas.dropna(inplace=True)
    datas['금리'] = 0.0
    datas.loc[((datas.index >= '2022-08-25')& (datas.index < '2022-10-12')), '금리'] = 2.5
    datas.loc[((datas.index >= '2022-10-12')& (datas.index < '2022-11-24')), '금리'] = 3.0
    datas.loc[((datas.index >= '2022-11-24')& (datas.index < '2023-01-13')), '금리'] = 3.25
    datas.loc[((datas.index >= '2023-01-13')), '금리'] = 3.5
    return datas