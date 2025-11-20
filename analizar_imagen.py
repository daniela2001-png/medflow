import time
import traceback
from modelo import init_medflow_model, device

def analizar_imagen(imagen, tipo_analisis="Reporte Estructurado", idioma="Español"):
    """
    Analiza una imagen médica utilizando el modelo Med-GEMMA.

    Procesa la imagen de entrada junto con un prompt basado en el tipo de análisis y idioma
    seleccionados. Genera un reporte médico estructurado o descriptivo.

    Args:
        imagen (PIL.Image.Image): La imagen médica a analizar.
        tipo_analisis (str, optional): El tipo de análisis a realizar.
            Puede ser "Descripción General", "Hallazgos Patológicos",
            "Reporte Estructurado", o "Diagnóstico Diferencial".
            Por defecto es "Reporte Estructurado".
        idioma (str, optional): El idioma en el que se generará el reporte.
            Puede ser "Español" o "Inglés". Por defecto es "Español".

    Returns:
        tuple: Una tupla conteniendo:
            - str: El reporte médico generado o un mensaje de error.
            - str: Metadatos del procesamiento (tiempo, modelo, GPU, etc.).
            - str: El estado del proceso (Completado, Error).
    """
    # Cargar el procesador y el modelo utilizando la función init_medflow_model
    processor, model = init_medflow_model()

    # Verificar si el modelo y procesador se cargaron correctamente
    if processor is None or model is None:
        print("🔴 No se pudo cargar el modelo MedFlow. Verifique los mensajes de error anteriores.")
    else:
        print("🟢 Modelo MedFlow listo para usar.")

    # Validar si se ha cargado una imagen
    if imagen is None:
        return "❌ Por favor carga una imagen primero", "", "Error: Sin imagen"

    try:
        inicio = time.time()

        # Prompts en español para los diferentes tipos de análisis
        prompts = {
            "Descripción General": "Describe esta imagen médica identificando las estructuras anatómicas visibles.",
            "Hallazgos Patológicos": "Identifica cualquier hallazgo patológico o anormal en esta imagen médica.",
            "Reporte Estructurado": "Genera un reporte médico estructurado con: TÉCNICA, HALLAZGOS e IMPRESIÓN.",
            "Diagnóstico Diferencial": "Proporciona un diagnóstico diferencial basado en los hallazgos visibles."
        }

        # Obtener el prompt adecuado según el tipo de análisis, con fallback a "Reporte Estructurado"
        prompt = prompts.get(tipo_analisis, prompts["Reporte Estructurado"])

        # Preparar los mensajes en el formato de chat para el modelo
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Eres un radiólogo experto especializado en interpretación de imágenes médicas."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": imagen}
                ]
            }
        ]

        # Aplicar el template de chat y tokenizar la entrada
        # Primero aplicar template sin tokenizar para obtener el texto completo
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # Importante: primero sin tokenizar
        )

        # Luego tokenizar por separado incluyendo la imagen
        inputs = processor(
            text=text_inputs,
            images=imagen,
            return_tensors="pt",
            padding=True
        )

        # Mover los tensores de entrada al dispositivo de procesamiento (GPU o CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Obtener la longitud de los tokens de entrada para decodificar solo la respuesta
        input_len = inputs["input_ids"].shape[-1]

        # Generar la respuesta del modelo
        print(f"🤖 Generando reporte...")

        # Usar torch.no_grad() para deshabilitar el cálculo de gradientes durante la inferencia
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,  # Número máximo de tokens a generar
                do_sample=False,  # Deshabilitar muestreo para una salida determinística
                num_beams=1, # Usar beam search con 1 haz (equivalente a greedy search)
                pad_token_id=processor.tokenizer.pad_token_id, # ID del token de padding
                eos_token_id=processor.tokenizer.eos_token_id # ID del token de fin de secuencia
            )

        # Decodificar solo los tokens generados por el modelo (excluyendo los tokens de entrada)
        generated_tokens = outputs[0][input_len:]
        reporte = processor.decode(generated_tokens, skip_special_tokens=True)

        tiempo = time.time() - inicio

        # Agregar un disclaimer médico al reporte
        disclaimer = """

⚠️ DISCLAIMER MÉDICO:
Este reporte es generado por IA con propósito educativo y demostrativo únicamente.
NO debe utilizarse para decisiones clínicas sin validación por profesionales médicos.
Proyecto académico - Corporación Universitaria Iberoamericana.
"""

        reporte_final = reporte + disclaimer

        # Generar metadatos del procesamiento
        metadata = f"""
📊 **Información de Procesamiento:**
- ⏱️ Tiempo: {tiempo:.2f} segundos
- 🤖 Modelo: Med-GEMMA 4B (Google Health AI)
- 💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- 📝 Tokens generados: {len(generated_tokens)}
- 🔧 Tipo análisis: {tipo_analisis}
"""

        # Establecer el estado de completado
        status = f"✅ Completado exitosamente en {tiempo:.2f}s"

        return reporte_final, metadata, status

    except Exception as e:
        # Capturar y formatear cualquier error que ocurra durante el procesamiento
        error_msg = f"""
❌ ERROR durante el análisis:

{str(e)}

**Posibles soluciones:**
1. Verifica que tengas GPU habilitada (Runtime > Change runtime type)
2. Reinicia el runtime (Runtime > Restart runtime)
3. Intenta con una imagen más pequeña
4. Si persiste, puede ser límite de memoria - prueba cerrar otras pestañas
"""
        print(f"\n❌ Error completo:\n{traceback.format_exc()}")
        # Retornar mensajes de error y estado
        return error_msg, "Error en procesamiento", "❌ Error"