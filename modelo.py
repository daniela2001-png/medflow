import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("No se encontró HF_TOKEN. Revisa tus secrets en Streamlit Cloud.")

def validate_gpu_type() -> bool:
    print("\n🔍 Verificando GPU...")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        return torch.cuda.is_available()
    else:
        print("⚠️ No hay GPU disponible")
        return False

is_cuda_setted = validate_gpu_type()
MODEL_ID = "google/medgemma-4b-it"
device = "cuda" if is_cuda_setted else "cpu"

def init_medflow_model():
    print("\n🏥 Iniciando MedFlow...")
    print(f"⚙️ Dispositivo: {device}")
    print("📥 Descargando modelo (puede tardar varios minutos)...\n")

    try:
        # CAMBIO: Pasar el token directamente aquí
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=hf_token
        )

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print("✅ Modelo cargado exitosamente!\n")
        return processor, model

    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        traceback.print_exc()
        return None, None