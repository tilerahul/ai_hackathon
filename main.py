from cProfile import label

import streamlit as st
import geonamescache
import datetime
st.title('Welcome to the Investment')
name=st.text_input("Enter Your name:")
st.write("Hii ", name)
email=st.text_input("E-Mail:")
mob=st.text_input("Mobile NO")
d = st.date_input("Select Date of Birth", datetime.date(2019, 7, 6))
st.write("Your birthday is:", d)
genre = st.radio(
    "Select Your Gender",
    ["Male", "Female","Others"],)
marr=st.radio(
    "Select Marrital Status",
    ["Married","Unmarried"],
)

label("Select Following")
states_dict = dict()
cs = ["Select a city"]

col1, col2, col3 = st.columns(3)

with col2:
    gc = geonamescache.GeonamesCache()
    city = gc.get_cities()
    states = gc.get_us_states()

    if "disabled" not in st.session_state:
        st.session_state.disabled = True


    def callback():
        st.session_state.disabled = False


    # ---STATES---

    for state, code in states.items():
        states_dict[code['name']] = state

    option_s = st.selectbox("Select a State", states_dict.items())

    if option_s == "Select a State":
        st.write("#")
    else:
        st.write("You selected:", option_s)

    # ---CITIES---

    for city in city.values():
        if city['countrycode'] == 'US':
            if city['admin1code'] == option_s[1]:
                cs.append(city['name'])

    option_c = st.selectbox("Select a city", cs, disabled=st.session_state.disabled, on_change=callback())

    if option_c == "Select a city":
        st.write("#")
    else:
        st.write("You selected:", option_c)

st.text("These Terms & Conditions (the “Terms”) apply to and regulate the provisions of investments products including mutual funds, collective.")
agree = st.checkbox("I agree")

st.link_button("Submit",url="detail.py")
if agree:
    st.write("Great!")