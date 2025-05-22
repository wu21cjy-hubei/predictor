
import streamlit as st
import numpy as np
import joblib

# 加载模型与标准化器
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Group Prediction with SVM")
st.write("请输入以下数值用于预测 group：")

# 用户输入
esr = st.number_input("ESR", min_value=0.0)
crp = st.number_input("CRP", min_value=0.0)
a_g = st.number_input("A/G", min_value=0.0)
wbc = st.number_input("WBC", min_value=0.0)
n_pct = st.number_input("Neutrophils (%)", min_value=0.0)
age = st.number_input("Age", min_value=0.0)
time_elapsed = st.number_input("Time elapsed to diagnosis (months)", min_value=0.0)
back_pain = st.selectbox("Back pain", options=[0, 1])
fever = st.selectbox("Fever", options=[0, 1])

# 组织特征
input_data = np.array([[esr, crp, a_g, wbc, n_pct, age, time_elapsed, back_pain, fever]])
input_scaled = scaler.transform(input_data)

# 预测
if st.button("预测 Group"):
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    st.success(f"预测的 Group 是: {prediction}")
    st.write("各组预测概率：")
    for i, prob in enumerate(probabilities, start=1):
        st.write(f"Group {i}: {prob:.2%}")

