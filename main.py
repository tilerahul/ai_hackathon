import pandas as pd
import babel.numbers
import plotly.graph_objects as go
# from cProfile import label

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

import streamlit as st
import numpy as np
st.title("Welcome to AI Based Allocation Engine" )
salary=st.text_input("Enter Your Name:")
genre = st.radio(
    "Select Type Of Employeement",
    ["Permanent", "Contract Base"],)

age=st.text_input("Enter Your Age:")
salary=st.text_input("Enter Your Salary:")
genre = st.radio(
    "Select Your Gender",
    ["Male", "Female","Others"],)
st.write('<style>div.row-widget.stRadio> div{flex-direction:row;}</style>',unsafe_allow_html=True)
amout=st.text_input("Enter the Amount You want to invest:")

sip_amount= st.slider("What Percentage of salary Do you Want to Invest", min_value = 10, max_value = 100)


duration= st.slider("Select The Investment Period", min_value = 10, max_value = 100)
rate_of_return = st.slider('Expected Rate of Return (in %)',1,30,12)

option = st.selectbox(
    "Risk Tolerance:",
    ("High", "Medium", "Low"))

st.write("You selected:", option)
checkbox = st.checkbox('', False)

# if inflation checkbox if off
if checkbox == False:

    monthly_rate = rate_of_return / 12 / 100
    months = duration * 12

    invested_value = sip_amount * months
    invested_value_inwords = babel.numbers.format_currency(invested_value, 'INR', locale='en_IN')

    future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
    future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

    gain = round(float(future_value) - float(invested_value), 2)
    gain_inwords = babel.numbers.format_currency(gain, 'INR', locale='en_IN')

    st.subheader(f'Amount Invested: {invested_value_inwords}')
    st.subheader(f'Final Amount: {future_value_inwords}')
    st.subheader(f'Gain: {gain_inwords}')

    # plot pie chart
    fig = go.Figure(data=[go.Pie(labels=['Investment', 'Gain'], values=[invested_value, gain])])
    fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                      marker=dict(colors=['green', 'red'], line=dict(color='#000000', width=2)))
    st.plotly_chart(fig)

elif checkbox == True:

    try:
        monthly_rate = (rate_of_return - 6) / 12 / 100
        months = duration * 12

        invested_value = sip_amount * months
        invested_value_inwords = babel.numbers.format_currency(sip_amount * months, 'INR', locale='en_IN')

        future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
        future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

        gain_after_inflation = round(float(future_value) - float(invested_value), 2)
        gain_after_inflation_inwords = babel.numbers.format_currency(gain_after_inflation, 'INR', locale='en_IN')

        st.subheader(f'Amount Invested: {invested_value_inwords}')
        st.subheader(f'Final Amount: {future_value_inwords}')
        st.subheader(f'Gain: {gain_after_inflation_inwords}')

        fig = go.Figure(data=[go.Pie(labels=['Investment', 'Gain'], values=[invested_value, gain_after_inflation])])
        fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                          marker=dict(colors=['green', 'red'], line=dict(color='#000000', width=2)))
        st.plotly_chart(fig)
    except Exception as e:
        st.warning('Please change the expcted rate of return')
st.button("Invest Now")

