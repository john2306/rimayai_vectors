# app/main.py
import torch
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import List, Union
from fastapi import FastAPI, HTTPException
from transformers import CLIPProcessor, CLIPModel

# --- Configuración del Modelo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
# Cargar el modelo y el procesador una sola vez al iniciar la aplicación
try:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"✅ CLIP Model ({model_name}) cargado en {device}.")
except Exception as e:
    print(f"❌ Error al cargar el modelo CLIP: {e}")
    # En un entorno Orin, si CUDA falla, forzamos la CPU para que la app se inicie.
    device = "cpu"
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
                img = Image.open(BytesIO(image_bytes))
                images.append(img)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error en imagen Base64: {e}")

    # Si no hay entradas, retornar error
    if not texts and not images:
        raise HTTPException(status_code=400, detail="Debe proporcionar texto o imagen(es).")
    
    # 2. Generación de Embeddings BATCH
    
    # Combinar texto e imagen para el procesamiento unificado de CLIP
    inputs = processor(text=texts if texts else None, 
                       images=images if images else None, 
                       return_tensors="pt", 
                       padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = []
    
    # 3. Formatear y Recopilar Resultados
    
    # Embeddings de Texto
    if texts:
        text_embeddings = outputs.text_embeds.cpu().numpy().tolist()
        for i, emb in enumerate(text_embeddings):
            results.append({"type": "text", "original": texts[i], "vector": emb})
            
    # Embeddings de Imagen
    if images:
        image_embeddings = outputs.image_embeds.cpu().numpy().tolist()
        # Aseguramos el orden de las imágenes en los resultados (aquí solo se muestra el índice)
        for i, emb in enumerate(image_embeddings):
            results.append({"type": "image", "original_index": i, "vector": emb})

    return {"status": "success", "count": len(results), "embeddings": results}

# Para probar con curl, puedes usar: 
# curl -X POST "http://localhost:8000/vectorize/batch" -H "Content-Type: application/json" -d '{"items": [{"text": "un gato"}, {"text": "un perro"}]}'