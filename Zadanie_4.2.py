import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()

# Load the trained model
filename = "model.sv"
model = pickle.load(open(filename, 'rb'))

# Dictionaries for categorical encoding
sex_d = {0: "Kobieta", 1: "Mężczyzna"}
chest_pain_d = {0: "Typ 1", 1: "Typ 2", 2: "Typ 3", 3: "Typ 4"}
resting_ecg_d = {0: "Normalny", 1: "Nieprawidłowość ST", 2: "Lewy blok odnogi pęczka Hisa"}
exercise_angina_d = {0: "Nie", 1: "Tak"}
st_slope_d = {0: "Opadający", 1: "Płaski", 2: "Wzrastający"}

# Reverse dictionaries for encoding user input
reverse_resting_ecg_d = {v: k for k, v in resting_ecg_d.items()}

def main():
    st.set_page_config(page_title="Predykcja chorób serca")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("images/heart.jpg", caption="Predykcja chorób serca", use_container_width=True)

    with overview:
        st.title("Predykcja ryzyka chorób serca")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        chest_pain_radio = st.radio("Typ bólu w klatce piersiowej", list(chest_pain_d.keys()), format_func=lambda x: chest_pain_d[x])
        exercise_angina_radio = st.radio("Dusznica podczas wysiłku", list(exercise_angina_d.keys()), format_func=lambda x: exercise_angina_d[x])
        st_slope_radio = st.radio("Nachylenie ST", list(st_slope_d.keys()), format_func=lambda x: st_slope_d[x])

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=20, max_value=80)
        resting_bp_slider = st.slider("Ciśnienie krwi w spoczynku", min_value=80, max_value=200, step=1, value=120)
        cholesterol_slider = st.slider("Cholesterol", min_value=100, max_value=400, step=1, value=200)
        fasting_bs_radio = st.radio("Cukier na czczo > 120 mg/dl", [0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
        max_hr_slider = st.slider("Maksymalne tętno", min_value=60, max_value=220, step=1, value=150)
        oldpeak_slider = st.slider("Depresja ST", min_value=0.0, max_value=6.0, step=0.1, value=1.5)

    # Prepare input data for the model
    data = [[
        age_slider,
        sex_radio,
        chest_pain_radio,
        resting_bp_slider,
        cholesterol_slider,
        fasting_bs_radio,
        reverse_resting_ecg_d["Normalny"],  # Default value, replace if needed
        max_hr_slider,
        exercise_angina_radio,
        oldpeak_slider,
        st_slope_radio
    ]]

    # Make predictions
    prediction_result = model.predict(data)
    prediction_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy dana osoba jest zagrożona chorobą serca?")
        st.subheader(("Tak" if prediction_result[0] == 1 else "Nie"))
        st.write("Pewność predykcji: {:.2f} %".format(prediction_confidence[0][prediction_result][0] * 100))

if __name__ == "__main__":
    main()
