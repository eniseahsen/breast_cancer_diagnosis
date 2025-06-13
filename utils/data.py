import pandas as pd
import streamlit as st
#veri çağırma ve preprocessing fonksiyonu
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    return df
