from cProfile import label

import streamlit as st
import numpy as np
st.title("Welcome to AI Based Allocation Engine", )
genre = st.radio(
    "Select Type Of Employeement",
    ["Permanent", "Contract Base"],)

age=st.text_input("Enter Your Age : ")
salary=st.text_input("Enter Your Salary:")
genre = st.radio(
    "Select Your Gender",
    ["Male", "Female","Others"],)
st.write('<style>div.row-widget.stRadio> div{flex-direction:row;}</style>',unsafe_allow_html=True)
amout=st.text_input("Enter the Amount You want to invest:")

per_salary= st.slider("What Percentage of salary Do you Want to Invest", min_value = 10, max_value = 100)

st.write(per_salary)
period = st.slider("Select The Investment Period", min_value = 10, max_value = 100)

st.write(period)

st.button("Submit")