import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import os
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)
_stop_words_es = set(stopwords.words('spanish'))
_tokenizer = RegexpTokenizer(r'\w+')
_stemmer = SnowballStemmer("spanish")

def text_preprocess(text):
    tokens = _tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in _stop_words_es]
    tokens = [_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Ruta absoluta basada en la ubicación de este archivo para evitar errores de directorio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'model', 'miniproject2-model.joblib')

model = joblib.load(model_path)

ods_nombres = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas"
}

st.title('Mini Proyecto 2 - Clasificacion de textos en ODS')

text_input = st.text_area("Ingrese un texto")

if st.button('Clasificar'):
    if text_input:
        with st.spinner("Clasificando..."):
            # El pipeline ya ejecuta text_preprocess internamente, 
            # solo necesitamos pasarle el texto en crudo
            prediction = model.predict([text_input])
            ods_num = prediction[0]
            ods_name = ods_nombres.get(ods_num, "Desconocido")
            st.success(f"El texto pertenece al **ODS {ods_num}: {ods_name}**")
    else:
        st.error("Por favor, ingrese un texto")