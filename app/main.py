# app/main.py
import torch
import base64
import os
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import List, Union
from fastapi import FastAPI, HTTPException
from transformers import CLIPProcessor, CLIPModel
import gc

# --- Configuración de Optimización para Jetson Orin ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.benchmark = True  # Optimiza convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # Usa TensorFloat32 para mayor velocidad

# --- Configuración del Modelo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
use_fp16 = torch.cuda.is_available()  # FP16 para optimizar memoria y velocidad en GPU

print(f"🔧 Dispositivo: {device}")
if device == "cuda":
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔧 CUDA Version: {torch.version.cuda}")
    print(f"🔧 Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cargar el modelo y el procesador una sola vez al iniciar la aplicación
try:
    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16 if use_fp16 else torch.float32)
    model = model.to(device)
    model.eval()  # Modo evaluación (desactiva dropout)
    
    # Compilar modelo para mayor velocidad (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Modelo compilado con torch.compile para mayor velocidad")
        except Exception as e:
            print(f"⚠️ No se pudo compilar el modelo: {e}")
    
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"✅ CLIP Model ({model_name}) cargado en {device} {'(FP16)' if use_fp16 else '(FP32)'}")
    
    # Liberar memoria
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        
except Exception as e:
    print(f"❌ Error al cargar el modelo CLIP: {e}")
    device = "cpu"
    use_fp16 = False
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"⚠️ El modelo se cargó en CPU. Revisar la configuración CUDA.")

app = FastAPI(title="Vectorizador Multimodal CLIP")

# --- Modelos de Pydantic para la API ---
class Item(BaseModel):
    """Define una entrada individual que puede ser texto o una imagen codificada en base64."""
    # El texto es opcional si se proporciona una imagen
    text: Union[str, None] = None
    # La imagen es una cadena base64 de la imagen, opcional si se proporciona texto
    image_b64: Union[str, None] = None

class BatchRequest(BaseModel):
    """Define la solicitud por lotes (BATCH) para la API."""
    items: List[Item]

# --- Endpoint Principal para Vectorización ---
@app.post("/vectorize/batch")
async def vectorize_batch(batch: BatchRequest):
    texts: List[str] = []
    images: List[Image.Image] = []
    
    # 1. Pre-procesamiento de las entradas (Parsing y Decodificación)
    for item in batch.items:
        if item.text:
            texts.append(item.text)
        
        if item.image_b64:
            try:
                # Decodificar Base64 a bytes y luego a objeto PIL Image
                image_bytes = base64.b64decode(item.image_b64)
                img = Image.open(BytesIO(image_bytes)).convert('RGB')  # Asegurar formato RGB
                images.append(img)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error en imagen Base64: {e}")

    # Si no hay entradas, retornar error
    if not texts and not images:
        raise HTTPException(status_code=400, detail="Debe proporcionar texto o imagen(es).")
    
    # 2. Generación de Embeddings BATCH con optimizaciones
    results = []
    
    try:
        # Procesar textos con autocast para FP16
        if texts:
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        text_embeddings = model.get_text_features(**text_inputs)
                else:
                    text_embeddings = model.get_text_features(**text_inputs)
            
            # Normalizar embeddings para similaridad coseno
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.cpu().float().numpy().tolist()
            
            for i, emb in enumerate(text_embeddings):
                results.append({"type": "text", "original": texts[i], "vector": emb})
            
            # Liberar memoria
            del text_inputs, text_embeddings
        
        # Procesar imágenes con autocast para FP16
        if images:
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            
            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        image_embeddings = model.get_image_features(**image_inputs)
                else:
                    image_embeddings = model.get_image_features(**image_inputs)
            
            # Normalizar embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings.cpu().float().numpy().tolist()
            
            for i, emb in enumerate(image_embeddings):
                results.append({"type": "image", "original_index": i, "vector": emb})
            
            # Liberar memoria
            del image_inputs, image_embeddings
        
        # Limpieza de memoria GPU
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {"status": "success", "count": len(results), "embeddings": results}
        
    except Exception as e:
        # Limpieza en caso de error
        if device == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Error al procesar embeddings: {str(e)}")

# --- Endpoints de Salud y Métricas ---
@app.get("/")
async def root():
    """Endpoint raíz con información básica"""
    return {
        "service": "Vectorizador Multimodal CLIP",
        "version": "1.0.0",
        "model": model_name,
        "device": device,
        "precision": "FP16" if use_fp16 else "FP32",
        "endpoints": {
            "vectorize": "/vectorize/batch",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint para monitoreo"""
    health_status = {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }
    
    if device == "cuda":
        health_status["gpu_available"] = torch.cuda.is_available()
        health_status["gpu_name"] = torch.cuda.get_device_name(0)
        health_status["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1e6, 2)
        health_status["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1e6, 2)
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Endpoint de métricas para monitoreo"""
    metrics = {
        "model": model_name,
        "device": device,
        "precision": "FP16" if use_fp16 else "FP32",
        "torch_version": torch.__version__,
    }
    
    if device == "cuda":
        metrics["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "memory": {
                "allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1e6, 2),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 2)
            }
        }
    
    return metrics

# Para probar con curl, puedes usar: 
# curl -X POST "http://localhost:8000/vectorize/batch" -H "Content-Type: application/json" -d '{"items": [{"text": "un gato"}, {"text": "un perro"}]}'