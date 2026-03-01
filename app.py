#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="🏋️ Exercise Recommendation System", layout="wide")

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1 {
    color: #FF4B4B;
    text-align: center;
}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #ff1a1a;
}
section[data-testid="stSidebar"] {
    background-color: #161A22;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏋️ Exercise Recommendation System</h1>", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("merged_exercise_dataset.csv")

df = load_data()

# ===================== DATA CLEANING =====================
df = df.drop_duplicates().dropna()
df.columns = df.columns.str.lower().str.strip()

# ===================== FEATURE ENGINEERING =====================
df["combined_features"] = (
    df["target"].str.lower() + " " +
    df["equipment"].str.lower() + " " +
    df["bodypart"].str.lower()
)

# ===================== TF-IDF MODEL =====================
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df["combined_features"])
similarity = cosine_similarity(feature_matrix)

# ===================== RECOMMEND FUNCTION =====================
def recommend_exercises(exercise_name):
    exercise_name = exercise_name.lower()
    names = df["name"].str.lower().tolist()
    
    match = get_close_matches(exercise_name, names, n=1, cutoff=0.4)
    
    if not match:
        return []
    
    index = df[df["name"].str.lower() == match[0]].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommended = [(df.iloc[i[0]]["name"], round(i[1], 2)) for i in sorted_scores]
    
    return recommended

# ===================== SIDEBAR FILTER =====================
st.sidebar.header("Filter Options")

body_filter = st.sidebar.selectbox("Select Body Part", ["All"] + sorted(df["bodypart"].unique()))
equipment_filter = st.sidebar.selectbox("Select Equipment", ["All"] + sorted(df["equipment"].unique()))

filtered_df = df.copy()

if body_filter != "All":
    filtered_df = filtered_df[filtered_df["bodypart"] == body_filter]

if equipment_filter != "All":
    filtered_df = filtered_df[filtered_df["equipment"] == equipment_filter]

st.subheader("📋 Available Exercises")
st.dataframe(filtered_df[["name", "target", "bodypart", "equipment"]].head(20))

# ===================== SEARCH SECTION =====================
st.subheader("🔍 Search Exercise for Recommendations")

exercise_input = st.text_input("Enter Exercise Name")

if st.button("Recommend"):
    if exercise_input:
        results = recommend_exercises(exercise_input)
        
        if results:
            st.success("Top 5 Recommended Exercises")
            
            for name, score in results:
                st.markdown(f"""
                <div style='background-color:#1C1F26;
                            padding:15px;
                            border-radius:10px;
                            margin-bottom:10px;
                            border-left:5px solid #FF4B4B;'>
                    <h4 style='color:white;'>{name}</h4>
                    <p style='color:gray;'>Similarity Score: {score}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Exercise not found")


# In[ ]:




