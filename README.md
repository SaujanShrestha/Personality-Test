# Introvert or Extrovert Predictor

This is a simple Streamlit web app that predicts whether you are an introvert or extrovert based on your personal and social habits. The app uses a trained machine learning model (LightGBM) to make predictions.

## Features

- Predicts personality as Introvert or Extrovert
- Takes user input from sliders and dropdowns:
  - Time spent alone
  - Social event attendance
  - Going outside frequency
  - Friend circle size
  - Social media post frequency
  - Stage fear (Yes/No)
  - Drained after socializing (Yes/No)

## How to Run

1. Make sure you have Python installed.
2. Install required packages:

   pip install streamlit pandas numpy joblib

3. Run the app:

   streamlit run app.py

4. The app will open in your browser. Use the sidebar to input your behavior, and click “Predict Personality” to see the result.

## Files

- app.py – Main Streamlit app
- model/lgbm_model.pkl – Pre-trained LightGBM model (required)

## Output

The app will display your input and predict whether you are likely to be an Introvert or an Extrovert.
