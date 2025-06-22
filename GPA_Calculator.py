import streamlit as st
import pandas as pd

# Function to calculate Cumulative GPA
def calculate_cumulative_gpa(previous_gpa, previous_credits, current_gpa, current_credits):
    total_points = (previous_gpa * previous_credits) + (current_gpa * current_credits)
    total_credits = previous_credits + current_credits
    return total_points / total_credits

# Streamlit app
st.title("GPA Calculator")

tab1, tab2 = st.tabs(["Term GPA", "Cumulative GPA"])

with tab1:
    # Ask user for the number of subjects
    num_subjects = st.number_input("How many subjects?", min_value=1, step=1, value=5)

    # Dictionary to map grades to points
    grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}

    # Create columns for better organization
    col1, col2, col3 = st.columns(3)

    subjects = []
    for i in range(1, num_subjects + 1):
        with st.container():
            subject_name = col1.text_input(f"Subject {i}", key=f"subject_{i}")
            grade = col2.selectbox(f"Grade {i}", grade_points.keys(), key=f"grade_{i}")
            credits = col3.number_input(f"Credits {i}", min_value=1, step=1, key=f"credits_{i}")
            subjects.append((grade, credits))

    # Calculate GPA button
    if st.button("Calculate GPA"):
        total_points = sum(grade_points[grade] * credits for grade, credits in subjects)
        total_credits = sum(credits for _, credits in subjects)
        gpa = total_points / total_credits if total_credits > 0 else 0
        st.success(f"Your Term GPA is: {gpa:.2f}")

with tab2:
    st.header("Calculate Cumulative GPA")
    previous_gpa = st.number_input("Previous cumulative GPA", min_value=0.0, max_value=5.0, step=0.01)
    previous_credits = st.number_input("Total credit hours completed before current term", min_value=0, step=1)
    current_gpa = st.number_input("Current term GPA", min_value=0.0, max_value=5.0, step=0.01)
    current_credits = st.number_input("Credit hours for current term", min_value=0, step=1)
    if st.button("Calculate Cumulative GPA"):
        cumulative_gpa = calculate_cumulative_gpa(previous_gpa, previous_credits, current_gpa, current_credits)
        st.success(f"Your Cumulative GPA is: {cumulative_gpa:.2f}")