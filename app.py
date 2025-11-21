# app.py
import gradio as gr
import requests
import os
from PIL import Image
import io
import time

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/medgemma-4b-it"

def analizar_imagen_api(imagen, tipo_analisis="Reporte Estructurado", idioma="Español"):
    if imagen is None:
        return "❌ Por favor carga una imagen primero", "", "Error: Sin imagen"
    
    try:
        inicio = time.time()
        
        print(f"📤 Enviando imagen a la API de Hugging Face...")
        
        # Convertir imagen a bytes
        buffered = io.BytesIO()
        imagen.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        prompts = {
            "Descripción General": "Describe esta imagen médica identificando las estructuras anatómicas visibles.",
            "Hallazgos Patológicos": "Identifica cualquier hallazgo patológico o anormal en esta imagen médica.",
            "Reporte Estructurado": "Genera un reporte médico estructurado con: TÉCNICA, HALLAZGOS e IMPRESIÓN.",
            "Diagnóstico Diferencial": "Proporciona un diagnóstico diferencial basado en los hallazgos visibles."
        }
        
        prompt = prompts.get(tipo_analisis, prompts["Reporte Estructurado"])
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Preparar payload (la API puede requerir formato específico)
        # Intentamos con la imagen directamente primero
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=img_bytes,
            timeout=120
        )
        
        tiempo_parcial = time.time() - inicio
        
        if response.status_code == 200:
            try:
                resultado = response.json()
                print(f"✅ Respuesta recibida: {resultado}")
                
                # Extraer texto de la respuesta
                if isinstance(resultado, list) and len(resultado) > 0:
                    if isinstance(resultado[0], dict):
                        reporte = resultado[0].get('generated_text', str(resultado))
                    else:
                        reporte = str(resultado[0])
                elif isinstance(resultado, dict):
                    reporte = resultado.get('generated_text', str(resultado))
                else:
                    reporte = str(resultado)
                
            except:
                reporte = response.text
            
            tiempo = time.time() - inicio
            
            disclaimer = """

⚠️ DISCLAIMER MÉDICO:
Este reporte es generado por IA con propósito educativo y demostrativo únicamente.
NO debe utilizarse para decisiones clínicas sin validación por profesionales médicos.
"""
            
            reporte_final = reporte + disclaimer
            
            metadata = f"""
📊 **Información de Procesamiento:**
- ⏱️ Tiempo total: {tiempo:.2f} segundos
- 🤖 Modelo: Med-GEMMA 4B (Google Health AI)
- 💻 Procesamiento: Hugging Face Inference API con GPU
- 🌐 Tu Space: CPU basic (solo interfaz)
- 🔧 Tipo análisis: {tipo_analisis}
"""
            
            status = f"✅ Completado en {tiempo:.2f}s"
            return reporte_final, metadata, status
        
        elif response.status_code == 503:
            return """
⏳ **El modelo se está cargando en los servidores de Hugging Face**

Por favor espera 20-30 segundos e intenta de nuevo.

(Esto solo pasa la primera vez o después de inactividad)
""", "", "🔄 Modelo cargando..."
        
        elif response.status_code == 401:
            return """
❌ **Error de autenticación**

Tu token HF_TOKEN no es válido o no tiene permisos.

Verifica:
1. Que el token existe en Settings → Secrets
2. Que aceptaste los términos en https://huggingface.co/google/medgemma-4b-it
""", "", "❌ Error de autenticación"
        
        else:
            error_detail = f"""
❌ **Error de la API de Hugging Face**

Código: {response.status_code}
Respuesta: {response.text[:500]}

Intenta de nuevo en unos segundos.
"""
            print(f"Error API: {response.status_code} - {response.text}")
            return error_detail, "", "❌ Error"
    
    except requests.exceptions.Timeout:
        return """
⏱️ **Timeout**

La solicitud tomó demasiado tiempo. Esto puede pasar si:
- El modelo está cargándose por primera vez
- Hay mucha demanda en los servidores

Por favor intenta de nuevo.
""", "", "⏱️ Timeout"
    
    except Exception as e:
        error_msg = f"""
❌ **Error durante el análisis:**

{str(e)}

Revisa los logs del Space para más detalles.
"""
        print(f"❌ Error completo: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", "❌ Error"

# Crear interfaz Gradio
css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}
h1 {
    text-align: center;
    color: #2563eb;
}
"""

with gr.Blocks(title="MedFlow MVP", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("""
    # 🏥 MedFlow - Producto Mínimo Viable
    ## Sistema de Interpretación Automatizada de Imágenes Médicas
    
    **Desarrollado por:** Yeinmy Daniela Morales Barrera  
    **Modelo:** Med-GEMMA 4B (Google Health AI)  
    **Infraestructura:** Hugging Face Inference API
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Entrada de Datos")
            
            imagen_input = gr.Image(
                type="pil",
                label="🖼️ Imagen Médica (Rayos X, TC, etc.)",
                height=350
            )
            
            tipo_analisis = gr.Dropdown(
                choices=[
                    "Descripción General",
                    "Hallazgos Patológicos",
                    "Reporte Estructurado",
                    "Diagnóstico Diferencial"
                ],
                value="Reporte Estructurado",
                label="📋 Tipo de Análisis"
            )
            
            idioma = gr.Radio(
                choices=["Español", "Inglés"],
                value="Español",
                label="🌐 Idioma del Reporte"
            )
            
            with gr.Row():
                procesar_btn = gr.Button("🚀 Analizar Imagen", variant="primary", size="lg")
                limpiar_btn = gr.ClearButton(components=[imagen_input], value="🗑️ Limpiar", size="lg")
            
            gr.Markdown("""
            **Nota:** La primera solicitud puede tardar 30-60 segundos mientras el modelo se carga en los servidores.
            Las siguientes serán más rápidas (20-40 segundos).
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Resultado del Análisis")
            
            reporte_output = gr.Textbox(
                label="Reporte Médico Generado",
                lines=18,
                placeholder="El reporte aparecerá aquí después de procesar la imagen...",
                show_copy_button=True
            )
            
            status_output = gr.Textbox(
                label="Estado del Proceso",
                lines=1,
                interactive=False
            )
    
    with gr.Accordion("📊 Metadatos de Procesamiento", open=False):
        metadata_output = gr.Markdown()
    
    procesar_btn.click(
        fn=analizar_imagen_api,
        inputs=[imagen_input, tipo_analisis, idioma],
        outputs=[reporte_output, metadata_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 📞 Información del Proyecto
    
    **Contacto:** ymoral35@estudiante.ibero.edu.co  
    **Versión:** MVP 1.0 - API Edition
    
    *Proyecto académico - Corporación Universitaria Iberoamericana*
    """)

if __name__ == "__main__":
    demo.launch()