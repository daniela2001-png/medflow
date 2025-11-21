import streamlit as st
from PIL import Image

st.set_page_config(page_title="MedFlow MVP", layout="wide")
st.title("🏥 MedFlow - Producto Mínimo Viable")

# Intentar cargar el modelo y capturar errores
st.info("🔄 Cargando modelo... esto puede tardar algunos minutos")

try:
    with st.spinner("Importando módulos..."):
        from analizar_imagen import analizar_imagen
        from modelo import init_medflow_model, device
    
    with st.spinner("Cargando modelo Med-GEMMA 4B..."):
        @st.cache_resource
        def cargar_modelo():
            processor, model = init_medflow_model()
            if processor is None or model is None:
                raise Exception("El modelo retornó None - revisa tu token HF y acceso al modelo")
            return processor, model
        
        processor, model = cargar_modelo()
    
    st.success("✅ Modelo cargado exitosamente!")

except Exception as e:
    st.error("❌ ERROR AL CARGAR EL MODELO")
    st.error(str(e))
    
    with st.expander("Ver detalles técnicos del error"):
        import traceback
        st.code(traceback.format_exc())
    
    st.markdown("""
    ### Posibles soluciones:
    
    1. **Verifica tu token de Hugging Face:**
       - Ve a https://huggingface.co/settings/tokens
       - Crea un nuevo token tipo "Read" (no Fine-grained)
       - Actualiza el secret `HF_TOKEN` en Streamlit Cloud
    
    2. **Acepta los términos del modelo:**
       - Ve a https://huggingface.co/google/medgemma-4b-it
       - Haz clic en "Agree and access repository"
       - Espera unos minutos y vuelve a cargar la app
    
    3. **Revisa los logs de Streamlit Cloud:**
       - Manage app → Logs
       - Busca mensajes de error específicos
    """)
    
    st.stop()

# Resto de tu código UI
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
            reporte, meta, status = analizar_imagen(imagen, tipo_analisis, idioma, processor, model, device)
        st.success(status)
        st.text_area("Reporte Médico", value=reporte, height=300)
        st.markdown(meta)
    else:
        st.warning("Debes subir una imagen para analizar.")

st.markdown("---")
st.markdown("Desarrollado por Yeinmy Daniela Morales Barrera - MedFlow MVP")