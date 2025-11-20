"""
Importa las librerías y módulos necesarios para la aplicación MedFlow.

Se importan:
- torch: Para operaciones con tensores y manejo de GPU.
- AutoProcessor, AutoModelForImageTextToText de transformers: Para cargar el modelo y procesador de Hugging Face.
- Image de PIL (Pillow): Para el manejo de imágenes.
- time: Para medir el tiempo de procesamiento.
- traceback: Para obtener información detallada de errores.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from dotenv import load_dotenv
import traceback
from huggingface_hub import login


# Cargamos variables desde .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("No se encontró HF_TOKEN. Revisa tus secrets en Streamlit Cloud.")
login(token=hf_token)




def validate_gpu_type() -> bool:
  """
  Valida el tipo de GPU a usar dentro del entorno de Google Colab.

  Verifica si CUDA está disponible y, si es así, imprime información sobre la GPU.
  Limpia la caché de CUDA para liberar memoria.

  Args:
    None

  Returns:
    bool: True si CUDA está disponible, False en caso contrario.
  """
  print("\n🔍 Verificando GPU...")
  print(f"CUDA disponible: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
      print(f"GPU: {torch.cuda.get_device_name(0)}")
      print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
      # Limpiar cache CUDA
      torch.cuda.empty_cache()
      return torch.cuda.is_available()
  else:
      print("⚠️ No hay GPU. Ve a Runtime > Change runtime type > T4 GPU")
      return False


validate_gpu_type()
# Validar si se ha configurado CUDA (GPU)
is_cuda_setted = validate_gpu_type()
# Definir el ID del modelo a utilizar de Hugging Face
MODEL_ID = "google/medgemma-4b-it"
# Configurar el dispositivo de procesamiento ('cuda' si hay GPU, 'cpu' en caso contrario)
device = "cuda" if is_cuda_setted else "cpu"


def init_medflow_model():
  """
  Inicializa el modelo de MedFlow (Med-GEMMA 4B Multimodal).

  Descarga y carga el modelo y el procesador asociados desde Hugging Face.
  Configura el modelo para usar bfloat16 y device_map="auto" para optimizar el uso de memoria y recursos.

  Args:
    None

  Returns:
    tuple: Una tupla conteniendo el procesador y el modelo cargados.
           Retorna (None, None) si ocurre un error durante la carga.
  """

  print(f"\n🏥 Iniciando MedFlow...")
  print(f"⚙️ Dispositivo: {device}")
  print(f"📥 Descargando modelo (5-10 min primera vez)...\n")

  try:
      # Cargar el procesador asociado al modelo
      processor = AutoProcessor.from_pretrained(MODEL_ID)

      # Cargar el modelo pre-entrenado
      model = AutoModelForImageTextToText.from_pretrained(
          MODEL_ID,
          torch_dtype=torch.bfloat16,  # Usar bfloat16 para eficiencia y compatibilidad con GPU
          device_map="auto",  # Distribuir el modelo automáticamente entre dispositivos disponibles
          low_cpu_mem_usage=True # Reducir el uso de CPU al cargar el modelo
      )

      print("✅ Modelo cargado exitosamente!\n")
      return processor, model

  except Exception as e:
      print(f"❌ Error cargando modelo: {e}")
      traceback.print_exc()
      return None, None