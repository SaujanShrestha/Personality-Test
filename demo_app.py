import streamlit as st
import numpy as np
import onnxruntime as rt
import pandas as pd

# Title
st.title("🚢 Titanic Survival Prediction (LightGBM ONNX Model)")

# Sidebar for user input
st.sidebar.header("Passenger Information")

# User Inputs
pclass = st.sidebar.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.2)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])  # C = Cherbourg, Q = Queenstown, S = Southampton

# --- One-hot encoding to match training ---
sex_female = 1 if sex.lower() == "female" else 0
sex_male = 1 if sex.lower() == "male" else 0

embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# --- Final input as 10 features in correct order ---
input_data = pd.DataFrame([{
    "pclass": pclass,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "embarked_C": embarked_C,
    "embarked_Q": embarked_Q,
    "embarked_S": embarked_S,
    "sex_female": sex_female,
    "sex_male": sex_male
}])

# Convert to numpy float32
input_np = input_data.to_numpy().astype(np.float32)

# Load and run ONNX model
try:
    sess = rt.InferenceSession("model/lightgbm_model.onnx")
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: input_np})
    prediction = int(pred[0][0])

    # Output
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.success("🎉 This passenger **survived**.")
    else:
        st.error("❌ This passenger **did not survive**.")
except Exception as e:
    st.error(f"Model could not be loaded or run. Error: {e}")
