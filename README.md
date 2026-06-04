# 🎧 Sistema Inteligente de Recomendación Musical

Aplicación web desarrollada con Streamlit que utiliza un modelo de recomendación basado en contenido para sugerir canciones similares a partir de una canción seleccionada por el usuario.

## 🚀 Aplicación en línea

🔗 https://proyectodevalor-g4ht8yjsyxjwsmyvih6nof.streamlit.app/

## 📋 Descripción del proyecto

El sistema analiza diferentes características musicales de las canciones para identificar similitudes y generar recomendaciones personalizadas.

Entre las variables utilizadas se encuentran:

- Popularidad
- Danceability
- Energía
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duración
- Género musical
- Contenido explícito

La recomendación se realiza mediante un modelo de similitud coseno aplicado a una matriz de características musicales.

## ✨ Funcionalidades

- Selección de canciones desde un catálogo musical.
- Búsqueda por nombre de canción o artista.
- Recomendación automática de canciones similares.
- Opción para excluir canciones del mismo artista.
- Dashboard interactivo con visualizaciones.
- Clasificación emocional de canciones.
- Análisis de géneros musicales.
- Estadísticas de popularidad por año.

## 📊 Visualizaciones incluidas

- Top géneros musicales.
- Popularidad promedio por año.
- Relación entre energía y valence.
- Artistas con mayor número de canciones.
- Distribución de emociones musicales.

## 🛠️ Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Plotly

## 🧠 Modelo de recomendación

El proyecto implementa un sistema de recomendación basado en contenido (Content-Based Recommendation System).

Proceso utilizado:

1. Limpieza y transformación de datos.
2. Normalización de variables numéricas mediante MinMaxScaler.
3. Vectorización de géneros musicales mediante CountVectorizer.
4. Construcción de una matriz de características.
5. Cálculo de similitud utilizando Cosine Similarity.
6. Generación de recomendaciones según la canción seleccionada.

## 📂 Dataset

El proyecto utiliza un conjunto de datos de canciones de Spotify con información musical, popularidad, género y características de audio.

## 👩‍💻 Autor

Lucila Auora Hernández Zamudio

## 🌐 Acceso rápido

👉 https://proyectodevalor-g4ht8yjsyxjwsmyvih6nof.streamlit.app/
