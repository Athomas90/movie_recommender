import streamlit as st
from movie_recommender import preprocess_movies, assign_clusters, recommend_by_title, random_recommendations
import pandas as pd

@st.cache_data
def load_data():
    df, df_model, scaled, features = preprocess_movies("final_df.csv")
    df, model = assign_clusters(df, df_model, scaled, n_clusters=5)
    return df

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎥", layout="centered")
st.title("🎬 Movie Recommender")
st.write("Discover movie recommendations based on title similarity or explore randomly.")

df = load_data()

title = st.text_input("🎥 Enter a movie title:")
mode = st.radio("Select Recommendation Type:", ["Cluster-Based", "Random"])

if title:
    st.subheader("🎯 Recommendations:")
    if mode == "Cluster-Based":
        recs = recommend_by_title(title, df)
        for r in recs:
            st.markdown(f"- {r}")
    elif mode == "Random":
        recs = random_recommendations(df, exclude_title=title)
        for r in recs:
            st.markdown(f"- {r}")
