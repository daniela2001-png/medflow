import time
import traceback
import torch

def analizar_imagen(imagen, tipo_analisis, idioma, processor, model, device):
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
    # Verificar si el modelo y procesador se cargaron correctamente
    if processor is None or model is None:
        return "❌ Error: el modelo no se cargó (verifica tu token HF o la memoria)", "", "❌ Error"

    if imagen is None:
        return "❌ Por favor carga una imagen primero", "", "Error: Sin imagen"

    try:
        inicio = time.time()
        prompts = {
            "Descripción General": "Describe esta imagen médica identificando las estructuras anatómicas visibles.",
            "Hallazgos Patológicos": "Identifica cualquier hallazgo patológico o anormal en esta imagen médica.",
            "Reporte Estructurado": "Genera un reporte médico estructurado con: TÉCNICA, HALLAZGOS e IMPRESIÓN.",
            "Diagnóstico Diferencial": "Proporciona un diagnóstico diferencial basado en los hallazgos visibles."
        }
        prompt = prompts.get(tipo_analisis, prompts["Reporte Estructurado"])
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
        text_inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text_inputs,
            images=imagen,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_len:]
        reporte = processor.decode(generated_tokens, skip_special_tokens=True)
        tiempo = time.time() - inicio
        disclaimer = """

⚠️ DISCLAIMER MÉDICO:
Este reporte es generado por IA solo con propósito educativo y demostrativo.
NO debe usarse para decisiones clínicas sin validación profesional.
"""
        reporte_final = reporte + disclaimer
        metadata = f"""
📊 **Información de Procesamiento:**
- ⏱️ Tiempo: {tiempo:.2f} segundos
- 🤖 Modelo: Med-GEMMA 4B (Google Health AI)
- 💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- 📝 Tokens generados: {len(generated_tokens)}
- 🔧 Tipo análisis: {tipo_analisis}
"""
        status = f"✅ Completado en {tiempo:.2f}s"
        return reporte_final, metadata, status
    except Exception as e:
        error_msg = f"""
❌ ERROR durante el análisis:

{str(e)}

**Posibles soluciones:**
1. Verifica tu token HuggingFace y acceso al modelo.
2. Reinicia la app si persiste.
3. Intenta con una imagen más pequeña o con menos carga en el sistema.
"""
        print(f"\n❌ Error completo:\n{traceback.format_exc()}")
        return error_msg, "Error en procesamiento", "❌ Error"