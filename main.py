##Se disponibilizan los datos usando FastAPI

from typing import Union
from fastapi import FastAPI
import pandas as pd
import numpy as np
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()



#ingestamos el Data Frame

df = pd.read_csv('df_arreglado.csv' )
copia_df=df.copy()
copia_df = copia_df.drop(df.index[:40376])
tfidf=TfidfVectorizer(stop_words='english')
copia_df['overview']=copia_df['overview'].fillna('')
tfidf_matrix=tfidf.fit_transform(copia_df['overview'])
tfidf_matrix.shape
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# cargamos las funciones ya revisadas

#API 1
@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes: str) -> dict:
    months_translated= {
    'enero': 'January',
    'febrero': 'February',
    'marzo': 'March',
    'abril': 'April',
    'mayo': 'May',
    'junio': 'June',
    'julio': 'July',
    'agosto': 'August',
    'septiembre': 'September',
    'octubre': 'October',
    'noviembre': 'November',
    'diciembre': 'December'}  
    fechas = pd.to_datetime(df['release_date'], format= '%Y-%m-%d')
    n_mes = fechas[fechas.dt.strftime('%B').str.capitalize() == months_translated[str(mes).lower()]]
    respuesta = n_mes.shape[0]
    return {'mes':mes, 'cantidad':respuesta}

#API 2
@app.get('/peliculas_dia{mes}')
def peliculas_dia(dia:str):
    fechas=pd.to_datetime(df['release_date'],format='%Y-%m-%d')
    ndia = fechas[fechas.dt.day_name(locale='es_AR') == dia.capitalize()]
    respuesta=ndia.shape[0]
    return{'dia':dia, 'cantidad':respuesta}

#API 3
@app.get('/franquicia{franquicia}')
def franquicia(franquicia:str):
    df.belongs_to_collection=df.belongs_to_collection.str.lower()
    m2=df[["belongs_to_collection","revenue"]]
    if isinstance(franquicia,str):
        franquicia=franquicia.lower()
        franquicia=unicodedata.normalize('NFKD',franquicia).encode(
            'ascii','ignore').decode('utf-8','ignore')
        
        ganancias=m2['revenue'][m2['belongs_to_collection'].str.contains(franquicia)==True]
        respuesta1=ganancias.shape[0]
        respuesta2=ganancias.sum()
        respuesta3=ganancias.mean()
    
    return{'franquicias':franquicia,'cantidad':respuesta1,'ganancia_total':respuesta2,'ganancia_promedio':respuesta3}

#API 4
@app.get('/pelicula_pais{pais}')
def peliculas_pais(pais):
    m1=df[["title", "production_countries"]]
    if isinstance(pais, str):
        pais=pais.lower()
        pais=unicodedata.normalize('NFKD', pais).encode(
            'ascii','ignore').decode('utf-8','ignore')
        cantidad=m1['title']
        [m1['production_countries'].str.contains(pais)==True]
        cantidad=df['production_countries'].apply(lambda x :str(x).lower()).map(str.lower).apply(lambda x:pais in x)
        respuesta= cantidad.shape [0]
        return{'pais': pais, 'cantidad': respuesta}
    
#API5 
@app.get('/productoras{productora}')
def productoras(productora:str):
    
    prod=df[['production_companies','budget','revenue']].dropna()
    prod['production_companies']=prod['production_companies'].map(str.lower)
    prod=prod[prod.production_companies.str.contains(productora.lower(),regex=False)]
    cantidad=prod.shape[0]
    gan_tot=(prod['revenue']-prod['budget']).sum()
    return{'productora': productora, 'ganancia_total':gan_tot, 'cantidad':cantidad}

# API 6

@app.get('/retorno{pelicula}')
def retorno(pelicula: str) -> dict:
    pelicula_df = df.loc[df['title'] == pelicula.title()]
    inversion = pelicula_df['budget'].iloc[0].item()
    ganancia = pelicula_df['revenue'].iloc[0].item()
    retorno= pelicula_df['return'].iloc[0].item()
    anio = pelicula_df['release_year'].iloc[0].item()
    return {'pelicula': pelicula, 'inversion': inversion, 'ganancia': ganancia, 'retorno': retorno, 'anio': anio }

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo):
    idx = indices[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recommendations=list(copia_df['title'].iloc[movie_indices].str.title())
    return {'lista recomendada': recommendations}