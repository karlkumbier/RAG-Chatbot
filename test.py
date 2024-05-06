import streamlit as st
import time

options = ("male", "female")

a = st.empty()

value = a.radio("gender", options, 0, key='a')

if value == 'male':
    st.write(value)
elif value == 'female':
    st.write(value)
    time.sleep(1)
    a.radio("gender", options, 0, key='b')
    st.write(value) 