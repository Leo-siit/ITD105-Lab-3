import streamlit as st
import numpy as np
import joblib

def load_model(file):
    try:
        model = joblib.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Interactive Prediction App")

    st.sidebar.header("Upload Model")
    model_file = st.sidebar.file_uploader("Upload your trained model (.joblib)", type=['joblib'])
    
    if model_file:
        model = load_model(model_file)
        if model:
            st.sidebar.success("Model loaded successfully!")
            
            # Create tabs for different prediction types
            heart_tab, co2_tab = st.tabs(["Heart Disease Prediction", "CO2 Concentration Prediction"])
            
            with heart_tab:
                st.header("Heart Disease Prediction")
                age = st.number_input("Age", min_value=0, max_value=120, value=0, key="heart_age")
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female', key="heart_sex")
                chest_pain_type = st.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4, value=1, key="heart_cp")
                bp = st.number_input("Blood Pressure", min_value=0, max_value=300, value=120, key="heart_bp")
                cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200, key="heart_chol")
                fbs_over_120 = st.selectbox("FBS > 120 mg/dl", options=[0, 1], key="heart_fbs")
                ekg_results = st.number_input("Resting EKG Results", min_value=0, max_value=2, value=0, key="heart_ekg")
                max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=100, key="heart_hr")
                exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], key="heart_angina")
                st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, key="heart_st")
                slope_of_st = st.number_input("Slope of ST Segment", min_value=0, max_value=3, value=2, key="heart_slope")
                num_vessels_fluro = st.number_input("Number of Major Vessels (Fluoroscopy)", min_value=0, max_value=4, value=0, key="heart_vessels")
                thallium = st.number_input("Thallium Stress Test Result", min_value=0, max_value=7, value=3, key="heart_thallium")

                heart_input = np.array([[age, sex, chest_pain_type, bp, cholesterol, fbs_over_120, 
                                        ekg_results, max_hr, exercise_angina, st_depression, 
                                        slope_of_st, num_vessels_fluro, thallium]])

                if st.button("Predict Heart Disease", key="heart_predict"):
                    prediction = model.predict(heart_input)
                    st.subheader("Prediction Result")
                    if prediction[0] == 0:
                        st.write("The predicted result is: **No Heart Disease**")
                    else:
                        st.write("The predicted result is: **Heart Disease is present**")

            with co2_tab:
                st.header("CO2 Concentration Prediction")
                year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="co2_year")
                month = st.number_input("Month", min_value=1, max_value=12, value=1, key="co2_month")
                day = st.number_input("Day", min_value=1, max_value=31, value=1, key="co2_day")
                cycle = st.number_input("CO2 Seasonal Cycle", min_value=300.0, max_value=500.0, value=400.0, key="co2_cycle")

                co2_input = np.array([[year, month, day, cycle]])

                if st.button("Predict CO2 Concentration", key="co2_predict"):
                    prediction = model.predict(co2_input)
                    st.subheader("Prediction Result")
                    st.write(f"The predicted CO2 concentration is: **{prediction[0]:.2f} ppm**")
        else:
            st.sidebar.error("Failed to load the model.")

if __name__ == "__main__":
    main()