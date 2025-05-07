import streamlit as st
import pandas as pd
from movie_recommender import recommend_by_title

# Load preprocessed clustered DataFrame
@st.cache_data
def load_data():
    return pd.read_csv("final_df.csv")

# Streamlit page config
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎥", layout="centered")
st.title("🎬 Movie Recommender")
st.write("Discover movie recommendations based on movie title and production similarities.")

# Load data
df = load_data()

# Movie input
title = st.text_input("🎥 Enter a movie title:")

if title:
    st.subheader("🎯 Recommendations:")
    recs = recommend_by_title(title, df)
    for r in recs:
        st.markdown(f"- {r}")
