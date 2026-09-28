import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/titanic_best_model.pkl"
IMAGE_PATH = "images/titanic.jpg"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

if os.path.exists(IMAGE_PATH):
    st.image(
        IMAGE_PATH,
        use_container_width=True
    )

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

if st.button("Predict Survival"):

    input_data = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked]
        }
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    survival_probability = probability[1] * 100
    non_survival_probability = probability[0] * 100

    if prediction == 1:
        st.success("Passenger is predicted to SURVIVE.")
    else:
        st.error("Passenger is predicted NOT TO SURVIVE.")

    st.write(
        f"Survival Probability: {survival_probability:.2f}%"
    )

    st.write(
        f"Non-Survival Probability: {non_survival_probability:.2f}%"
    )