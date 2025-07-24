import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("model/lgbm_model.pkl")

# Title
st.title("Are you an introvert or extrovert?")

# Sidebar for user input
st.sidebar.header("Personal Information")

Time_spent_Alone = st.sidebar.slider("How many hours you spent alone?", 0, 11, 0)
Social_event_attendance = st.sidebar.slider("How often do you attent social events?", 0, 10, 0)
Going_outside = st.sidebar.slider("How often do you go outside?", 0, 7, 0)
Friends_circle_size = st.sidebar.slider("What is your friend circle size?", 0, 15, 0)
Post_frequency = st.sidebar.slider("how many times do you post in social media?", 0, 10, 0)
Stage_fear = st.sidebar.selectbox("Stage Fear", options=["No", "Yes"])
Drained_after_socializing = st.sidebar.selectbox("Drained After Socializing", options=["No", "Yes"])



Stage_fear = 1 if Stage_fear == "Yes" else 0
Drained_after_socializing = 1 if Drained_after_socializing == "Yes" else 0

input_df = pd.DataFrame([[
    Time_spent_Alone,
    Stage_fear,
    Social_event_attendance,
    Going_outside,
    Drained_after_socializing,
    Friends_circle_size,
    Post_frequency
]], columns=[
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
])

if st.button("Predict Personality"):
    st.write("Input data:", input_df)
    pred = model.predict(input_df)[0]
    label = "Extrovert" if pred == 1 else "Introvert"
    st.success(f"Predicted Personality: {label}")
