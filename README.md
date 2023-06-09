

#                   PROYECTO INDIVIDUAL 
#               MACHINE LEARNING OPERATIONS

# Introduccion y Objetivos:
Se nos entrego una base de datos que contenia informacion sobre peliculas, año de estreno,dia de estreno,tematica , pais de origen y demas cosas asociadas.
En base a ello debimos armar un sistema de recomendacion utilizando las herramientas disponibles para ello, donde, dada una pelicula,el sistema nos devuelva nombres de otros films que traten sobre el  mismo tema.

# Tareas desarrolladas:

- ETL : Extraccion y limpieza de la Base de Datos que se recibio
- Tareas encomendadas: eliminacion de columnas, desagrupamiento de datos , etc
- Generacion de las funciones para las API (Interfaz de Programacion de Aplicaciones). Deployment
- Realizacion del EDA: Analisis Exploratorio de Datos
- Modelo de Recomendacion 

- Se solicito:
-    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se        estrenaron ese mes historicamente'''
        '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia historicamente'''
        '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
       '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
        '''Ingresas la productora, retornando la ganancia toal y la cantidad de peliculas que produjeron'''
        '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo'''
   
# ETL y Tareas encomendadas:
Al explorar la Base nos dimos cuenta que los datos entregados estaban organizados en dos columnas con diccionarios que a su vez en un caso tenian listas dentro,ademas de ello se encontraron tres columnas formadas por listas.
Se procedio al desanidado de esas columnas (unnest) utilizando el modulo AST (Abstract Syntaxis Tree) , que ayuda a procesar árboles de la gramática de sintaxis abstracta de Python. 
Por lo demas, se realizaron las tareas de eliminar nulos, columnas duplicadas y agregado de nuevas mas un analisis de la Base en si misma.

# Generacion de las funciones para las API-Deployment:
Se desarrollaron las funciones en Python (Jupyter Notebook) con la consignas solicitadas.
Después de instalar FastAPI (web framework para construir APIs con Python 3.6+)  y uvicorn ( una implementación de servidor ASGI, que maneja las solicitudes de forma asincronica,  para Python), se creó un archivo "main.py" con la estructura necesaria para implementar los puntos finales en base a las funciones  construidas.
Se corrio localmente en el servicio uvicorn en el puerto 8000.

# EDA
Se hicieron dos analisis:
uno global  con la libreria "matplotib" para ver en forma general la cantidad de nulos por columna y uno particular extrayendo las columnas que nos servirian para el modelo.
Este ultimo analisis se hizo con la libreria "ydata_profiling".
# Conclusiones:
se observo que el modelo debe ser no supervisado y se utilizo entonces para su implementacion  CountVectorizer para calcular la matriz de similitud del coseno. En función de las puntuaciones de similitud proporcionadas por la matriz, el algoritmo recomienda las 5 películas más similares a la proporcionada por el usuario como entrada.
Esta ultima funcion tambien se puede ver desarrollada en el ultimo punto de las API.

# OBSERVACIONES:  
en el desarrollo del modelo se encontraron problemas de recursos que en su mayoria pudieron ser subsanados.
Debido a ello  para la implementacion final con CountVectorizer hubo que reducir la Base de Datos a 5000 registros (eran 45376 originalmente) para que se vea que el modelo funciona (ver main.py) aunque las recomendaciones del mismo no sean las logicas debido a que la Base de Datos se debio reducir a mas de un 50%.

# Nota final:
Quisiera agradecer aqui  a las personas que me ayudaron a llevar a cabo este proyecto , el primero en mi carrera de Data Science:
Profesores de Henry, en especial nuestros TA's Ricardo y Marcos
Compañeros , todos ellos , con una muy especial consideracion a Ariel,Angel y Karla,siempre alli y a toda hora para ayudarme con mis errores y alentarme a seguir adelante

# Video y Links

https://www.youtube.com/watch?v=Uup2MO_itxI

https://peliculas-t7ss.onrender.com






