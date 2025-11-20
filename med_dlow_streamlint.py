import streamlit as st
from PIL import Image
import torch
from analizar_imagen import analizar_imagen
from modelo import init_medflow_model, device

@st.cache_resource
def cargar_modelo():
    processor, model = init_medflow_model()
    return processor, model

processor, model = cargar_modelo()

st.set_page_config(page_title="MedFlow MVP", layout="wide")
st.title("🏥 MedFlow - Producto Mínimo Viable")
st.markdown("Automatiza la interpretación de imágenes médicas con IA.")

img_file = st.file_uploader("Sube una imagen médica (JPG, PNG)", type=["jpg", "jpeg", "png"])
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
            # Le PASAS processor, model, device:
            reporte, meta, status = analizar_imagen(imagen, tipo_analisis, idioma, processor, model, device)
        st.success(status)
        st.text_area("Reporte Médico", value=reporte, height=300)
        st.markdown(meta)
    else:
        st.warning("Debes subir una imagen para analizar.")
else:
    st.info("Sube una imagen y presiona 'Analizar Imagen' para empezar.")

st.markdown("---")
st.markdown("Desarrollado por Yeinmy Daniela Morales - MedFlow MVP")