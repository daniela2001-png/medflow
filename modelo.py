import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from dotenv import load_dotenv
import traceback
from huggingface_hub import login, HfApi

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# ===== VERIFICACIÓN DE TOKEN =====
print("🔍 DEBUG - Verificando token...")
if hf_token:
    print(f"✅ Token encontrado: {hf_token[:10]}...{hf_token[-4:]}")
    print(f"📏 Longitud del token: {len(hf_token)}")
    
    # Prueba de acceso al Hub
    try:
        api = HfApi()
        whoami = api.whoami(token=hf_token)
        print(f"👤 Usuario autenticado: {whoami['name']}")
        
        # Intenta verificar acceso al modelo específico
        try:
            model_info = api.model_info("google/medgemma-4b-it", token=hf_token)
            print(f"✅ Acceso confirmado al modelo: {model_info.id}")
        except Exception as e:
            print(f"⚠️ No se puede acceder al modelo: {e}")
            print("Ve a https://huggingface.co/google/medgemma-4b-it y acepta los términos")
    except Exception as e:
        print(f"❌ Error verificando usuario: {e}")
else:
    print("❌ Token NO encontrado")
    raise RuntimeError("No se encontró HF_TOKEN. Revisa tus secrets en Streamlit Cloud.")

# Login en Hugging Face Hub
try:
    login(token=hf_token, add_to_git_credential=False)
    print("✅ Login exitoso en Hugging Face Hub")
except Exception as e:
    print(f"⚠️ Advertencia en login: {e}")
# ==================================

def validate_gpu_type() -> bool:
    print("\n🔍 Verificando GPU...")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        return torch.cuda.is_available()
    else:
        print("⚠️ No hay GPU disponible, usando CPU")
        return False

is_cuda_setted = validate_gpu_type()
MODEL_ID = "google/medgemma-4b-it"
device = "cuda" if is_cuda_setted else "cpu"

def init_medflow_model():
    print("\n🏥 Iniciando MedFlow...")
    print(f"⚙️ Dispositivo: {device}")
    print(f"🎯 Modelo: {MODEL_ID}")
    print("📥 Descargando modelo (esto puede tardar 5-10 minutos la primera vez)...\n")

    try:
        # Intenta cargar el processor
        print("📦 [1/2] Cargando processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=hf_token,
            trust_remote_code=True
        )
        print("✅ Processor cargado correctamente")

        # Intenta cargar el modelo
        print("📦 [2/2] Cargando modelo (puede tardar varios minutos)...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            token=hf_token,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ Modelo cargado exitosamente!\n")
        
        return processor, model

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO AL CARGAR MODELO:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print("\n📋 Traceback completo:")
        traceback.print_exc()
        
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("\n🔐 ERROR DE AUTENTICACIÓN:")
            print("1. Ve a: https://huggingface.co/google/medgemma-4b-it")
            print("2. Asegúrate de haber aceptado los términos haciendo clic en 'Agree and access repository'")
            print("3. Crea un nuevo token tipo 'Read' (no Fine-grained)")
            print("4. Actualiza el secret HF_TOKEN en Streamlit Cloud")
        elif "403" in error_str or "forbidden" in error_str:
            print("\n🚫 ERROR DE PERMISOS:")
            print("Tu token no tiene permisos para acceder a este modelo gated")
            print("Solución: Crea un token nuevo con permisos 'Read'")
        elif "404" in error_str:
            print("\n🔍 ERROR: MODELO NO ENCONTRADO")
            print("El modelo no existe o no tienes acceso")
        else:
            print("\n❓ ERROR DESCONOCIDO")
            print("Revisa el traceback arriba para más detalles")
        
        return None, None