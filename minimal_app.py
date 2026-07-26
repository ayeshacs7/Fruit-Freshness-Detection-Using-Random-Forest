import streamlit as st

st.title("Test")

uploaded = st.file_uploader("Upload")

if uploaded is not None:
    st.success("Upload Successful")