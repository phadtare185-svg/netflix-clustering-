import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load model
with open(r"C:\Users\phadt\OneDrive\Desktop\MLproj\Netflix clustering\netflix_clustering_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

scaler = artifacts["scaler"]
mlb = artifacts["mlb"]
le = artifacts["label_encoder"]
kmeans = artifacts["kmeans"]
pca = artifacts["pca"]
feature_columns = artifacts["feature_columns"]

st.title("🎬 Netflix Show Clustering App")

# User Inputs
genres = st.multiselect(
    "Select Genres",
    options=mlb.classes_
)

rating = st.selectbox(
    "Select Rating",
    le.classes_
)

duration = st.slider(
    "Duration (minutes)",
    min_value=30,
    max_value=300,
    value=90
)

# Predict Button
if st.button("Predict Cluster"):
    # Encode genres
    genre_encoded = mlb.transform([genres])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    # Encode rating
    rating_encoded = le.transform([rating])[0]

    # Create input dataframe
    input_df = genre_df.copy()
    input_df["rating_encoded"] = rating_encoded
    input_df["duration_num"] = duration

    # Align features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Cluster: {cluster}")

    # PCA Visualization
    pca_point = pca.transform(input_scaled)

    fig, ax = plt.subplots()
    ax.scatter(pca_point[:,0], pca_point[:,1], c="red", s=100)
    ax.set_title("Your Show in PCA Space")
    st.pyplot(fig)
