import streamlit as st
from PIL import Image
import torch
from analizar_imagen import analizar_imagen
from modelo import init_medflow_model
# ... tus otros imports y tu código para inicializar el modelo y el processor ...

# Inicialización del modelo MedFlow (ajusta según tu código)
@st.cache_resource
def cargar_modelo():
    # Aquí va tu función de inicialización, por ejemplo:
    processor, model = init_medflow_model()
    return processor, model

processor, model = cargar_modelo()

st.set_page_config(page_title="MedFlow MVP", layout="wide")
st.title("🏥 MedFlow - Producto Mínimo Viable")
st.markdown("Automatiza la interpretación de imágenes médicas con IA.")

img_file = st.file_uploader(
    "Sube una imagen médica (JPG, PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Se aceptan imágenes radiológicas, dermatológicas, etc. Menos de 5MB."
)
tipo_analisis = st.selectbox(
    "Tipo de análisis",
    ["Descripción General", "Hallazgos Patológicos", "Reporte Estructurado", "Diagnóstico Diferencial"]
)
idioma = st.radio("Idioma del reporte", ("Español", "Inglés"))

if st.button("Analizar Imagen"):
    if img_file is not None:
        imagen = Image.open(img_file)
        st.image(imagen, caption="Imagen subida", use_column_width=True)
        with st.spinner("Procesando análisis..."):
            reporte, meta, status = analizar_imagen(imagen, tipo_analisis, idioma)
        st.success(status)
        st.text_area("Reporte Médico", value=reporte, height=300)
        st.markdown(meta)
    else:
        st.warning("Debes subir una imagen para analizar.")
else:
    st.info("Sube una imagen y presiona 'Analizar Imagen' para empezar.")

st.markdown("---")
st.markdown("Desarrollado por Yeinmy Daniela Morales - MedFlow MVP")