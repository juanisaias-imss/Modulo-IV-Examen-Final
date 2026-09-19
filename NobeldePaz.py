import streamlit as st
import os

st.write("Archivos encontrados:")

for archivo in os.listdir("."):
    st.write(archivo)
