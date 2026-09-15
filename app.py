import streamlit as st
import pandas as pd
import joblib
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Swastik✨")
st.markdown("Provide the following details")

# making UI
age = st.slider("Age" ,18 , 100 , 40)  #it creates a slideer for age set default at 40 with least val 18 and max 100
sex = st.selectbox("Sex" , ['M' , 'F']) # it is a select box
chest_pain = st.selectbox("Chest Pain Type" , ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)" , 80 , 200 , 120)
cholestrol = st.number_input("cholestrol(mg/dl)" , 100 , 600 , 200)
fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl" , [0 , 1])
resting_ecg = st.selectbox("resting ECG" , ['Normal' , 'ST' , 'LVH'])
max_hr = st.slider("max Heart Rate" , 60 , 220 , 150)
exercise_angina = st.selectbox("Excercise-Induced Angina" , ['Y' , 'N'])
oldpeak = st.slider("Oldpeak(St Depression)" , 0.0 , 6.0 , 1.0)
st_slope = st.selectbox("ST slope" , ['Up' , 'Flat' , 'Down'])

if st.button("predict"):
    raw_input = {                 # the dataframe
        'Age' : age,
        'RestingBP' : resting_bp,
        'Cholesterol' : cholestrol,
        'FastingBS' : fasting_bs,
        'MaxHR' : max_hr,
        'Oldpeak' : oldpeak,
        'Sex_' + sex : 1,
        'ChestPainType_' + chest_pain : 1,
        'RestingECG_' + resting_ecg : 1,
        'ExerciseAngina_' + exercise_angina : 1,
        'ST_Slope_' + st_slope : 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    input_df = input_df[expected_columns]  # reorder/fill as you already do
    # scale only numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")