import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

st.write('# Nobel Prize category prediction')

st.image(
    "NobeldePaz.png",
    caption="Its creator was the Swedish inventor Alfred Nobel through his will in 1895."
)

st.header('Motivation')

# Entrada de usuario
def user_input_features():

    texto = st.text_input("Enter the text to be evaluated")

    user_input_data = {
        'Motivation': texto
    }

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df = user_input_features()

# Cargar datos
nobel = pd.read_csv("nobel_consolidado.csv", encoding='utf-8')

# Codificación numérica
nobel['label_num'] = nobel['Category'].map({
    'chemistry': 0,
    'economics': 1,
    'literature': 2,
    'medicine': 3,
    'peace': 4,
    'physics': 5
})

# Variables
X = nobel['Motivation']
y = nobel['label_num']

# Vectorización
vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

# Modelo
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_dtm, y)

# Predicción solo si se captura texto
if len(df['Motivation'][0]) > 0:

    df_dtm = vect.transform(df['Motivation'])

    prediction = rf.predict(df_dtm)

    pred = prediction[0]

    st.subheader('Predicción')

    if pred == 0:
        st.write('Chemistry')
    elif pred == 1:
        st.write('Economics')
    elif pred == 2:
        st.write('Literature')
    elif pred == 3:
        st.write('Medicine')
    elif pred == 4:
        st.write('Peace')
    elif pred == 5:
        st.write('Physics')
    else:
        st.write('Sin predicción')
