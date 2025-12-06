#!/usr/bin/env python3
"""
Ejemplos de uso del API Vectorizador CLIP
"""

import requests
import base64
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def example_1_texto_simple():
    """Ejemplo 1: Vectorizar texto simple"""
    print("=" * 60)
    print("Ejemplo 1: Vectorizar texto simple")
    print("=" * 60)
    
    payload = {
        "items": [
            {"text": "un gato negro"},
            {"text": "un perro grande"},
            {"text": "una casa azul"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/vectorize/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"📊 Total embeddings: {data['count']}")
        print(f"📏 Dimensión del vector: {len(data['embeddings'][0]['vector'])}")
        
        # Mostrar primeros valores
        for emb in data['embeddings']:
            print(f"\nTexto: '{emb['original']}'")
            print(f"Vector (primeros 5): {emb['vector'][:5]}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def example_2_imagen_base64():
    """Ejemplo 2: Vectorizar imagen desde archivo"""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Vectorizar imagen (base64)")
    print("=" * 60)
    
    # Crear una imagen de ejemplo simple (si no tienes una)
    # Necesitarías tener una imagen real. Aquí un placeholder:
    print("ℹ️  Para este ejemplo necesitas una imagen.")
    print("   Puedes usar el siguiente código con tu imagen:")
    print("""
    from PIL import Image
    import io
    
    # Cargar tu imagen
    img = Image.open("mi_imagen.jpg")
    
    # Convertir a base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Enviar al API
    payload = {"items": [{"image_b64": img_b64}]}
    response = requests.post(f"{BASE_URL}/vectorize/batch", json=payload)
    """)


def example_3_mixto():
    """Ejemplo 3: Vectorizar texto e imagen juntos"""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Batch mixto (texto + imagen)")
    print("=" * 60)
    
    payload = {
        "items": [
            {"text": "un gato"},
            {"text": "un perro"},
            # {"image_b64": "..."}, # Agregar tu imagen aquí
        ]
    }
    
    response = requests.post(f"{BASE_URL}/vectorize/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Procesados: {data['count']} items")
        
        for emb in data['embeddings']:
            print(f"\n{emb['type'].upper()}: ", end="")
            if emb['type'] == 'text':
                print(f"'{emb['original']}'")
            else:
                print(f"Índice {emb['original_index']}")


def example_4_similaridad():
    """Ejemplo 4: Calcular similaridad entre textos"""
    print("\n" + "=" * 60)
    print("Ejemplo 4: Similaridad coseno entre textos")
    print("=" * 60)
    
    import numpy as np
    
    payload = {
        "items": [
            {"text": "un gato negro"},
            {"text": "un felino oscuro"},
            {"text": "un automóvil rojo"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/vectorize/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extraer vectores
        vectors = [np.array(emb['vector']) for emb in data['embeddings']]
        texts = [emb['original'] for emb in data['embeddings']]
        
        # Los vectores ya están normalizados, similaridad = dot product
        print("\nMatriz de similaridad:")
        print("-" * 60)
        
        for i, text1 in enumerate(texts):
            print(f"\n'{text1}':")
            for j, text2 in enumerate(texts):
                if i != j:
                    similarity = np.dot(vectors[i], vectors[j])
                    print(f"  vs '{text2}': {similarity:.4f}")


def example_5_health_check():
    """Ejemplo 5: Verificar salud del servicio"""
    print("\n" + "=" * 60)
    print("Ejemplo 5: Health check y métricas")
    print("=" * 60)
    
    # Root endpoint
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("\n📋 Información del servicio:")
        print(json.dumps(response.json(), indent=2))
    
    # Health check
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("\n💚 Health check:")
        print(json.dumps(response.json(), indent=2))
    
    # Metrics
    response = requests.get(f"{BASE_URL}/metrics")
    if response.status_code == 200:
        print("\n📊 Métricas del sistema:")
        print(json.dumps(response.json(), indent=2))


def example_6_curl_commands():
    """Ejemplo 6: Comandos curl equivalentes"""
    print("\n" + "=" * 60)
    print("Ejemplo 6: Comandos curl")
    print("=" * 60)
    
    print("\n1️⃣ Vectorizar texto:")
    print("""
curl -X POST "http://localhost:8000/vectorize/batch" \\
  -H "Content-Type: application/json" \\
  -d '{"items": [{"text": "un gato"}, {"text": "un perro"}]}'
    """)
    
    print("\n2️⃣ Health check:")
    print("""
curl http://localhost:8000/health
    """)
    
    print("\n3️⃣ Métricas:")
    print("""
curl http://localhost:8000/metrics
    """)


if __name__ == "__main__":
    print("🚀 Ejemplos de Uso - API Vectorizador CLIP")
    print("=" * 60)
    
    try:
        # Verificar que el servicio está disponible
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Servicio disponible\n")
            
            # Ejecutar ejemplos
            example_1_texto_simple()
            example_2_imagen_base64()
            example_3_mixto()
            example_4_similaridad()
            example_5_health_check()
            example_6_curl_commands()
            
            print("\n" + "=" * 60)
            print("✅ Todos los ejemplos completados!")
            print("=" * 60)
            print(f"\n📖 Documentación interactiva: {BASE_URL}/docs")
            
        else:
            print(f"❌ Servicio respondió con código: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ No se puede conectar a {BASE_URL}")
        print("   Verifica que el servicio esté corriendo:")
        print("   docker-compose up -d")
    except Exception as e:
        print(f"❌ Error: {e}")
