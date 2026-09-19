import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

st.title("Nobel Prize Category Prediction")

st.image("NobeldePaz.png")

nobel = pd.read_csv("nobel_consolidado.csv")

X = nobel["Motivation"]
y = nobel["Category"]

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_dtm, y)

texto = st.text_input("Enter the text to be evaluated")

if texto:

    texto_dtm = vect.transform([texto])

    prediction = rf.predict(texto_dtm)

    st.subheader("Predicción")
    st.success(prediction[0])
