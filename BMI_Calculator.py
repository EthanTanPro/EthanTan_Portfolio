import streamlit as st

def calculate_bmi(weight, height):
    height_m = height / 100  # convert height to meters
    bmi = weight / (height_m ** 2)
    return bmi

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

st.title("BMI Calculator")

weight = st.number_input("Enter your weight (in kg):", min_value=0.0, format="%.2f")
height = st.number_input("Enter your height (in cm):", min_value=0.0, format="%.2f")

if st.button("Calculate BMI"):
    if weight > 0 and height > 0:
        bmi = calculate_bmi(weight, height)
        category = bmi_category(bmi)
        st.write(f"Your BMI is: {bmi:.2f}")
        color_map = {"Underweight": "purple", "Normal weight": "blue", "Overweight": "orange", "Obese": "red"}
        color = color_map.get(category, "black")
        st.markdown(f"<h4 style='color:{color};'>Health category: {category}</h4>", unsafe_allow_html=True)
    else:
        st.write("Please enter valid weight and height values.")
