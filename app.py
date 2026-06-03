import html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ---------------------------------------------------------

st.set_page_config(
    page_title="Sistema Inteligente de Recomendación Musical",
    page_icon="🎧",
    layout="wide"
)


# ---------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------

@st.cache_data
def cargar_datos():
    df = pd.read_csv("spotify2000a2019.csv")

    # Limpiar caracteres especiales en texto
    columnas_texto = ["artist", "song", "genre"]

    for columna in columnas_texto:
        df[columna] = df[columna].astype(str).apply(lambda x: html.unescape(x))

    # Eliminar duplicados usando artista y canción
    # Se conserva la versión con mayor popularidad
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["artist", "song"], keep="first")

    # Reemplazar géneros vacíos o incorrectos
    df["genre"] = df["genre"].replace("set()", "unknown")

    # Crear duración en minutos
    df["duration_min"] = df["duration_ms"] / 60000

    # Convertir explicit a número
    df["explicit_int"] = df["explicit"].astype(str).str.lower().map({
        "true": 1,
        "false": 0
    }).fillna(0)

    # Limpiar género para usarlo en el modelo
    df["genre_clean"] = (
        df["genre"]
        .astype(str)
        .str.lower()
        .str.replace("&", "and", regex=False)
        .str.replace("/", " ", regex=False)
        .str.replace(",", " ", regex=False)
    )

    # Clasificación emocional simple usando energy y valence
    def clasificar_emocion(row):
        energia = row["energy"]
        valence = row["valence"]

        if energia >= 0.65 and valence >= 0.60:
            return "Alegre / Energética"
        elif energia >= 0.65 and valence < 0.60:
            return "Intensa"
        elif energia < 0.65 and valence >= 0.60:
            return "Relajada / Positiva"
        else:
            return "Triste / Melancólica"

    df["emotion"] = df.apply(clasificar_emocion, axis=1)

    # Crear columna para mostrar canción y artista juntos
    df["opcion"] = df["artist"] + " - " + df["song"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO DE RECOMENDACIÓN
# ---------------------------------------------------------

@st.cache_resource
def construir_modelo(df):
    columnas_numericas = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_min",
        "explicit_int"
    ]

    # Normalizar variables numéricas
    scaler = MinMaxScaler()
    datos_numericos = scaler.fit_transform(df[columnas_numericas])

    # Convertir género musical a variables numéricas
    vectorizer = CountVectorizer()
    datos_genero = vectorizer.fit_transform(df["genre_clean"])

    # Unir datos numéricos y datos de género
    matriz_caracteristicas = hstack([datos_numericos, datos_genero])

    # Calcular similitud entre canciones
    matriz_similitud = cosine_similarity(matriz_caracteristicas)

    return matriz_similitud


# ---------------------------------------------------------
# FUNCIÓN PARA GENERAR RECOMENDACIONES
# ---------------------------------------------------------

def recomendar_canciones(df, matriz_similitud, indice_cancion, cantidad=10, excluir_mismo_artista=False):
    similitudes = list(enumerate(matriz_similitud[indice_cancion]))
    similitudes = sorted(similitudes, key=lambda x: x[1], reverse=True)

    artista_base = df.iloc[indice_cancion]["artist"]

    recomendaciones = []

    for indice, score in similitudes:
        if indice == indice_cancion:
            continue

        if excluir_mismo_artista and df.iloc[indice]["artist"] == artista_base:
            continue

        recomendaciones.append({
            "Artista": df.iloc[indice]["artist"],
            "Canción": df.iloc[indice]["song"],
            "Año": int(df.iloc[indice]["year"]),
            "Género": df.iloc[indice]["genre"],
            "Emoción": df.iloc[indice]["emotion"],
            "Popularidad": int(df.iloc[indice]["popularity"]),
            "Danceability": round(df.iloc[indice]["danceability"], 2),
            "Energía": round(df.iloc[indice]["energy"], 2),
            "Valence": round(df.iloc[indice]["valence"], 2),
            "Tempo": round(df.iloc[indice]["tempo"], 2),
            "Duración min": round(df.iloc[indice]["duration_min"], 2),
            "Similitud": round(score, 3)
        })

        if len(recomendaciones) >= cantidad:
            break

    return pd.DataFrame(recomendaciones)


# ---------------------------------------------------------
# CARGA DEL DATASET Y MODELO
# ---------------------------------------------------------

df = cargar_datos()
matriz_similitud = construir_modelo(df)


# ---------------------------------------------------------
# INTERFAZ PRINCIPAL
# ---------------------------------------------------------

st.title("🎧 Sistema Inteligente de Recomendación Musical")

st.markdown(
    """
    Este proyecto utiliza inteligencia artificial y análisis de datos para recomendar canciones 
    similares a partir de una canción seleccionada por el usuario. El sistema analiza características 
    como género, popularidad, energía, danceability, valence, tempo y duración.
    """
)


# ---------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------

st.sidebar.header("Configuración del sistema")

cantidad = st.sidebar.slider(
    "Número de recomendaciones",
    min_value=3,
    max_value=20,
    value=10
)

excluir_mismo_artista = st.sidebar.checkbox(
    "Excluir canciones del mismo artista",
    value=False
)

tipo_busqueda = st.sidebar.radio(
    "Tipo de búsqueda",
    ["Seleccionar canción", "Buscar por texto"]
)


# ---------------------------------------------------------
# SELECCIÓN DE CANCIÓN
# ---------------------------------------------------------

st.subheader("1. Selección de canción")

if tipo_busqueda == "Seleccionar canción":
    cancion_seleccionada = st.selectbox(
        "Selecciona una canción:",
        df["opcion"].sort_values().tolist()
    )

else:
    texto_busqueda = st.text_input("Escribe el nombre de una canción o artista:")

    if texto_busqueda:
        resultados = df[df["opcion"].str.contains(texto_busqueda, case=False, na=False)]

        if len(resultados) > 0:
            cancion_seleccionada = st.selectbox(
                "Resultados encontrados:",
                resultados["opcion"].sort_values().tolist()
            )
        else:
            st.warning("No se encontraron canciones con ese texto.")
            st.stop()
    else:
        st.info("Escribe una canción o artista para buscar.")
        st.stop()


indice_cancion = df[df["opcion"] == cancion_seleccionada].index[0]
cancion_base = df.iloc[indice_cancion]


# ---------------------------------------------------------
# INFORMACIÓN DE LA CANCIÓN SELECCIONADA
# ---------------------------------------------------------

st.subheader("2. Canción seleccionada")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Artista", cancion_base["artist"])

with col2:
    st.metric("Año", int(cancion_base["year"]))

with col3:
    st.metric("Popularidad", int(cancion_base["popularity"]))

with col4:
    st.metric("Duración", f"{round(cancion_base['duration_min'], 2)} min")

st.write("**Canción:**", cancion_base["song"])
st.write("**Género:**", cancion_base["genre"])
st.write("**Emoción estimada:**", cancion_base["emotion"])
st.write("**Danceability:**", round(cancion_base["danceability"], 2))
st.write("**Energía:**", round(cancion_base["energy"], 2))
st.write("**Valence:**", round(cancion_base["valence"], 2))
st.write("**Tempo:**", round(cancion_base["tempo"], 2))


# ---------------------------------------------------------
# RECOMENDACIONES
# ---------------------------------------------------------

st.subheader("3. Recomendaciones musicales")

recomendaciones = recomendar_canciones(
    df,
    matriz_similitud,
    indice_cancion,
    cantidad,
    excluir_mismo_artista
)

st.dataframe(recomendaciones, use_container_width=True)

st.markdown(
    """
    Las canciones recomendadas son aquellas que tienen características similares a la canción seleccionada.
    La columna **Similitud** indica qué tan parecida es cada canción respecto a la canción base.
    Mientras más cercano sea el valor a 1, mayor es la similitud.
    """
)


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.subheader("4. Dashboard de análisis musical")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Géneros",
    "Popularidad por año",
    "Energía vs Valence",
    "Top artistas",
    "Emociones"
])


# ---------------------------------------------------------
# GRÁFICA DE GÉNEROS
# ---------------------------------------------------------

with tab1:
    generos = df.assign(genre_split=df["genre"].str.split(",")).explode("genre_split")
    generos["genre_split"] = generos["genre_split"].str.strip()

    conteo_generos = generos["genre_split"].value_counts().reset_index()
    conteo_generos.columns = ["Género", "Cantidad"]

    fig_generos = px.bar(
        conteo_generos.head(15),
        x="Género",
        y="Cantidad",
        title="Top 15 géneros musicales en la base de datos",
        color="Cantidad"
    )

    st.plotly_chart(fig_generos, use_container_width=True)


# ---------------------------------------------------------
# GRÁFICA DE POPULARIDAD POR AÑO
# ---------------------------------------------------------

with tab2:
    popularidad_anual = df.groupby("year")["popularity"].mean().reset_index()

    fig_year = px.line(
        popularidad_anual,
        x="year",
        y="popularity",
        markers=True,
        title="Popularidad promedio por año"
    )

    fig_year.update_layout(
        xaxis_title="Año",
        yaxis_title="Popularidad promedio"
    )

    st.plotly_chart(fig_year, use_container_width=True)


# ---------------------------------------------------------
# GRÁFICA ENERGÍA VS VALENCE
# ---------------------------------------------------------

with tab3:
    fig_scatter = px.scatter(
        df,
        x="valence",
        y="energy",
        color="emotion",
        hover_data=["artist", "song", "genre", "popularity"],
        title="Relación entre valence y energía"
    )

    fig_scatter.update_layout(
        xaxis_title="Valence: positividad musical",
        yaxis_title="Energía"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------
# GRÁFICA TOP ARTISTAS
# ---------------------------------------------------------

with tab4:
    top_artistas = df["artist"].value_counts().head(15).reset_index()
    top_artistas.columns = ["Artista", "Cantidad de canciones"]

    fig_artistas = px.bar(
        top_artistas,
        x="Artista",
        y="Cantidad de canciones",
        title="Top 15 artistas con más canciones",
        color="Cantidad de canciones"
    )

    st.plotly_chart(fig_artistas, use_container_width=True)


# ---------------------------------------------------------
# GRÁFICA DE EMOCIONES
# ---------------------------------------------------------

with tab5:
    conteo_emociones = df["emotion"].value_counts().reset_index()
    conteo_emociones.columns = ["Emoción", "Cantidad"]

    fig_emociones = px.pie(
        conteo_emociones,
        names="Emoción",
        values="Cantidad",
        title="Distribución de canciones por emoción estimada"
    )

    st.plotly_chart(fig_emociones, use_container_width=True)


# ---------------------------------------------------------
# EXPLICACIÓN DEL MODELO
# ---------------------------------------------------------

st.subheader("5. Explicación del modelo")

st.markdown(
    """
    El sistema utiliza un modelo de recomendación basado en contenido. Este tipo de modelo compara 
    las características de cada canción y recomienda las canciones más similares a la seleccionada 
    por el usuario.

    Para construir el modelo se utilizaron variables como:

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
    - Si la canción es explícita o no

    Primero, los datos numéricos se normalizan para que estén en una misma escala. Después, el género 
    musical se convierte en datos numéricos. Finalmente, se calcula la similitud entre canciones usando 
    similitud coseno.

    La similitud coseno permite medir qué tan parecidas son dos canciones considerando varias 
    características al mismo tiempo.
    """
)


# ---------------------------------------------------------
# INFORMACIÓN GENERAL DEL DATASET
# ---------------------------------------------------------

st.subheader("6. Información general de la base de datos")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Total de canciones", len(df))

with col_b:
    st.metric("Año mínimo", int(df["year"].min()))

with col_c:
    st.metric("Año máximo", int(df["year"].max()))

with col_d:
    st.metric("Popularidad promedio", round(df["popularity"].mean(), 2))


st.markdown(
    """
    El proyecto demuestra cómo la inteligencia artificial puede aplicarse a una actividad cotidiana, 
    como escuchar música, para generar recomendaciones personalizadas y facilitar el análisis de datos.
    """
)
