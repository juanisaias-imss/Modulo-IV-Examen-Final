%%writefile NobeldePaz.py
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier



st.write(''' # Nobel Prize category prediction''')
st.image("NobeldePaz.png", caption="Its creator was the Swedish inventor Alfred Nobel through his will in 1895.")

st.header('Motivation')

def user_input_features():
  # Entrada
  texto = st.text_input("Enter the text to be evaluated")

  user_input_data = {'Motivation': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

url3 = "https://raw.githubusercontent.com/juanisaias-imss/Modulo-IV-Examen-Final/refs/heads/main/nobel_consolidado.csv"
nobel =  pd.read_csv(url3, encoding='utf-8')
X = nobel.Motivation
y = nobel.Category

vect = CountVectorizer()

X_dtm = vect.fit_transform(X)

#nb = MultinomialNB()
#nb.fit(X_dtm, y)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_dtm, y)

df_dtm = vect.transform(df['Motivation'])
prediction = rf.predict(df_dtm)

{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('Predicción')
if prediction == 0:
  st.write('Physics')
elif prediction == 1:
  st.write('Medicine')
elif prediction == 2:
  st.write('Peace')
elif prediction == 3:
  st.write('Literature')
elif prediction == 4:
  st.write('Chemistry')
elif prediction == 5:
  st.write('Economics')
else:
  st.write('Sin predicción')
