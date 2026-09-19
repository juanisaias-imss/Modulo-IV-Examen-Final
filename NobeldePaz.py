import streamlit as st
import subprocess

st.write("Inicio")

resultado = subprocess.run(
    ["pip", "list"],
    capture_output=True,
    text=True
)

st.text(resultado.stdout)
