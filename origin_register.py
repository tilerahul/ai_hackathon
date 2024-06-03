import pandas as pd
import babel.numbers
import plotly.graph_objects as go
from cProfile import label

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
# st.write(data)
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from scipy.optimize import minimize

data = pd.read_csv("employees_data.csv")
data = data.dropna()
# st.write(data)
data['Risk Tolerance'] = data['Risk Tolerance'].map({'Low':0, 'Medium':1, 'High':2})


x = data[['Age', 'Salary', 'Retirement Target', 'Monthly Investment']]
y = data['Risk Tolerance']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
# st.write(model.score(x_test, y_test)*100)
# st.write( model.score(x_train, y_train)*100)

market_data = {
    'investment_option': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'historical_return': [0.05, 0.10, 0.15, 0.08, 0.12, 0.20],
    'market_volatility': [0.02, 0.03, 0.05, 0.03, 0.04, 0.06],
    'risk_level': [1, 2, 3, 2, 3, 4]
}

df_market = pd.DataFrame(market_data)
# st.write(df_market)

X_market = df_market[['historical_return', 'market_volatility']]
y_market_return = df_market['historical_return']
y_market_risk = df_market['risk_level']

X_train_market, X_test_market, y_train_return, y_test_return, y_train_risk, y_test_risk = train_test_split(
    X_market, y_market_return, y_market_risk, test_size=0.2, random_state=42)

scaler_market = StandardScaler()
X_train_market = scaler_market.fit_transform(X_train_market)
X_test_market = scaler_market.transform(X_test_market)

# model_return = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train_market.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])


# model_return.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# model_return.fit(X_train_market, y_train_return, epochs=5, batch_size=10
# , validation_split=0.2)


# model_risk = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train_market.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])

# model_risk.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# model_risk.fit(X_train_market, y_train_risk, epochs=5, batch_size=10, validation_split=0.2)

model_return = load_model('model_return.h5')
model_risk = load_model('model_risk.h5')

def predict_investment_return_risk(investment_data):
    investment_data = scaler_market.transform([investment_data])
    predicted_return = model_return.predict(investment_data)[0][0]
    predicted_risk = model_risk.predict(investment_data)[0][0]
    return predicted_return, predicted_risk

expected_returns = []
expected_risks = []
for option in market_data['investment_option']:
    index = market_data['investment_option'].index(option)
    historical_return = market_data['historical_return'][index]
    market_volatility = market_data['market_volatility'][index]
    predicted_return, predicted_risk = predict_investment_return_risk([historical_return, market_volatility])
    expected_returns.append(predicted_return)
    expected_risks.append(predicted_risk)

expected_returns = np.array(expected_returns)
expected_risks = np.array(expected_risks)
# st.write(expected_returns)
# st.write(expected_risks)

def portfolio_optimization(expected_returns, expected_risks, risk_tolerance):
    n = len(expected_returns)
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.diag(expected_risks**2), weights)))
        return -portfolio_return + risk_tolerance * portfolio_risk

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = n * [1. / n,]

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


risk_tolerance = 0.5
optimal_weights = portfolio_optimization(expected_returns, expected_risks, risk_tolerance)*100
# st.write(optimal_weights)


st.title("Welcome to AI Based Allocation Engine" )
salary=st.text_input("Enter Your Name:")
genre = st.radio(
    "Select Type Of Employeement",
    ["Permanent", "Contract Base"],)

genre = st.radio(
    "Select Your Gender",
    ["Male", "Female","Others"],)

age=st.slider("Select Your Age", min_value = 18, max_value = 65)
salary=st.number_input("Enter Your Salary:")

st.write('<style>div.row-widget.stRadio> div{flex-direction:row;}</style>',unsafe_allow_html=True)

retirement_target =st.number_input("Enter the Retirement Target")

investment_amount =st.number_input("Enter the Amount You want to invest:")

investment_period = st.slider("Select The Investment Period", min_value = 1, max_value = 50)

st.write(model.predict([[age,salary,retirement_target,investment_amount]]))

# sip_amount= st.slider("What Percentage of salary Do you Want to Invest", min_value = 10, max_value = 100)


# duration= st.slider("Select The Investment Period", min_value = 10, max_value = 100)
# rate_of_return = st.slider('Expected Rate of Return (in %)',1,30,12)

# option = st.selectbox(
#     "Risk Tolerance:",
#     ("High", "Medium", "Low"))


# if inflation checkbox if off


# monthly_rate = rate_of_return / 12 / 100
# months = duration * 12

# invested_value = sip_amount * months
# invested_value_inwords = babel.numbers.format_currency(invested_value, 'INR', locale='en_IN')

# future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
# future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

# gain = round(float(future_value) - float(invested_value), 2)
# gain_inwords = babel.numbers.format_currency(gain, 'INR', locale='en_IN')

# st.subheader(f'Amount Invested: {invested_value_inwords}')
# st.subheader(f'Final Amount: {future_value_inwords}')
# st.subheader(f'Gain: {gain_inwords}')

# plot pie chart
fig = go.Figure(data=[go.Pie(labels=market_data['investment_option'], values=optimal_weights)])
fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15)
st.plotly_chart(fig)

# st.write(optimal_weights)


# try:
#     monthly_rate = (rate_of_return - 6) / 12 / 100
#     months = duration * 12

#     invested_value = sip_amount * months
#     invested_value_inwords = babel.numbers.format_currency(sip_amount * months, 'INR', locale='en_IN')

#     future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
#     future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

#     gain_after_inflation = round(float(future_value) - float(invested_value), 2)
#     gain_after_inflation_inwords = babel.numbers.format_currency(gain_after_inflation, 'INR', locale='en_IN')

#     st.subheader(f'Amount Invested: {invested_value_inwords}')
#     st.subheader(f'Final Amount: {future_value_inwords}')
#     st.subheader(f'Gain: {gain_after_inflation_inwords}')

#     fig = go.Figure(data=[go.Pie(labels=['Investment', 'Gain'], values=[invested_value, gain_after_inflation])])
#     fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
#                         marker=dict(colors=['green', 'red'], line=dict(color='#000000', width=2)))
#     st.plotly_chart(fig)
# except Exception as e:
#     st.warning('Please change the expcted rate of return')


st.button("Invest Now")

