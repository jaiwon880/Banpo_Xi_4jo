import pandas as pd
import streamlit as st
import glob

@st.cache_data
def read_data():
    glob_pt = '/app/jaiwon880/ml_projiect/web/static/*.csv'
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(glob_pt))])  


def get_unique_list(data, index):
    return data.str.split(expand=True)[index].unique()

def check_contain(data, kwd):
    return data.str.contains(kwd)

@st.cache_data
def get_city_list():
    df = read_data()
    return get_unique_list(df['시군구'], 0)

@st.cache_data
def get_gu_list(city_choice):
    df = read_data()
    return get_unique_list(df['시군구'][
        check_contain(df['시군구'], city_choice)], 1)

@st.cache_data
def get_town_list(city_choice, gu_choice):
    df = read_data()
    return get_unique_list(df['시군구'][
        check_contain(df['시군구'], city_choice)
        & check_contain(df['시군구'], gu_choice)], 2)

@st.cache_data
def get_village_list(city_choice, gu_choice, town_choice):
    df = read_data()
    filtered = df['시군구'][
        check_contain(df['시군구'], city_choice)
        & check_contain(df['시군구'], gu_choice)
        & check_contain(df['시군구'], town_choice)
        ]
    if len(filtered.str.split(expand=True).columns) < 4:
        return []
        
    return get_unique_list(filtered, 3)