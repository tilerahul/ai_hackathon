import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy.optimize import minimize

@st.cache_data
def load_employee_data():
    data = pd.read_csv("Employees_DataSet2.csv")
    data['Risk Tolerance'] = data['Risk Tolerance'].map({'Low': 0.25, 'Medium': 0.5, 'High': 0.75})
    return data

@st.cache_data
def preprocess_employee_data(data):
    x = data[['Age', 'Salary', 'Retirement Target', 'Monthly Investment']]
    y = data['Risk Tolerance']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test

@st.cache_resource
def train_employee_model(x_train, y_train):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(x_train, y_train)
    return model

@st.cache_data
def load_market_data():
    market_data = pd.read_csv('Stock_data.csv')
    market_data['Risk Level'] = market_data['Risk Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    return market_data

@st.cache_data
def preprocess_market_data(market_data):
    X_market = market_data[['Historical return', 'Market Volatility']]
    y_market_return = market_data['Historical return']
    y_market_risk = market_data['Risk Level']
    X_train_market, X_test_market, y_train_return, y_test_return, y_train_risk, y_test_risk = train_test_split(
        X_market, y_market_return, y_market_risk, test_size=0.2, random_state=42)
    scaler_market = StandardScaler()
    X_train_market = scaler_market.fit_transform(X_train_market)
    X_test_market = scaler_market.transform(X_test_market)
    return scaler_market, X_train_market, X_test_market, y_train_return, y_test_return, y_train_risk, y_test_risk

@st.cache_resource
def load_pretrained_models():
    model_return = load_model('model_return.h5')
    model_risk = load_model('model_risk.h5')
    return model_return, model_risk

def predict_investment_return_risk(investment_data, scaler_market, model_return, model_risk):
    investment_data = scaler_market.transform([investment_data])
    predicted_return = model_return.predict(investment_data)[0][0]
    predicted_risk = model_risk.predict(investment_data)[0][0]
    return predicted_return, predicted_risk

def portfolio_optimization(expected_returns, expected_risks, risk_tolerance):
    expected_returns = np.array(expected_returns)
    expected_risks = np.array(expected_risks)
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

# Load data and models
data = load_employee_data()
x_train, x_test, y_train, y_test = preprocess_employee_data(data)
employee_model = train_employee_model(x_train, y_train)

market_data = load_market_data()
scaler_market, X_train_market, X_test_market, y_train_return, y_test_return, y_train_risk, y_test_risk = preprocess_market_data(market_data)
model_return, model_risk = load_pretrained_models()

# Streamlit web application
st.title("Welcome to AI Based Allocation Engine")
name = st.text_input("Enter Your Name:")
employment_type = st.radio("Select Type Of Employment", ["Permanent", "Contract Base"])
gender = st.radio("Select Your Gender", ["Male", "Female", "Others"])
age = st.slider("Select Your Age", min_value=18, max_value=65)
salary = st.number_input("Enter Your Salary:")
st.write('<style>div.row-widget.stRadio> div{flex-direction:row;}</style>', unsafe_allow_html=True)
retirement_target = st.number_input("Enter the Retirement Target")
investment_amount = st.number_input("Enter the Amount You want to invest:")
investment_period = st.slider("Select The Investment Period", min_value=1, max_value=50)
submit = st.button("Submit")

if submit:
    risk_tolerance = employee_model.predict([[age, salary, retirement_target, investment_amount]])[0]  # Predict risk tolerance
    optimal_weights = portfolio_optimization(market_data['Expected Returns'], market_data['Expected Risks'], risk_tolerance) * 100

    top_6_indices = np.argsort(optimal_weights)[::-1][:6]
    selected_exp_return = [market_data['Expected Returns'][i] for i in top_6_indices]
    selected_exp_risk = [market_data['Expected Risks'][i] for i in top_6_indices]
    stock_name = [market_data['Name'][i] for i in top_6_indices]

    sel_optimal_weights = portfolio_optimization(selected_exp_return, selected_exp_risk, risk_tolerance) * 100

    # Plot pie chart
    fig = go.Figure(data=[go.Pie(labels=stock_name, values=sel_optimal_weights, texttemplate='%{label}: %{value:.2f}%')])
    fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15)
    st.plotly_chart(fig)
