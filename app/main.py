import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
_stop_words_es = set(stopwords.words('spanish'))
_tokenizer = RegexpTokenizer(r'\w+')
_stemmer = PorterStemmer()

def text_preprocess(text):
    tokens = _tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in _stop_words_es]
    tokens = [_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

model = joblib.load('../model/miniproject2-model.joblib')

st.title('Mini Proyecto 2 - Clasificacion de textos en objetivos de desarrollo sostenible')

text_input = st.text_area("Ingrese un texto")

if st.button('Clasificar'):
    if text_input:
        with st.spinner("Clasificando..."):
            prediction = model.predict([text_preprocess(text_input)])
            st.write(f"El texto pertenece al ODS {prediction[0]}")
    else:
        st.error("Por favor, ingrese un texto")