# Script de benchmark para evaluar rendimiento
import time
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

BASE_URL = "http://localhost:8000"

def test_text_batch(num_items=10):
    """Test de rendimiento con texto"""
    payload = {
        "items": [{"text": f"ejemplo de texto número {i}"} for i in range(num_items)]
    }
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/vectorize/batch", json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        return elapsed, num_items
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, 0

def benchmark():
    """Ejecutar benchmark completo"""
    print("🚀 Benchmark de Vectorizador CLIP en Jetson Orin Nano")
    print("=" * 60)
    
    # Warmup
    print("🔥 Warmup (3 requests)...")
    for _ in range(3):
        test_text_batch(5)
    
    # Tests con diferentes tamaños de batch
    batch_sizes = [1, 5, 10, 20, 50]
    results = []
    
    print("\n📊 Tests de rendimiento:")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        times = []
        for _ in range(5):  # 5 repeticiones por batch size
            elapsed, count = test_text_batch(batch_size)
            if elapsed:
                times.append(elapsed)
        
        if times:
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            results.append({
                "batch_size": batch_size,
                "avg_time": avg_time,
                "throughput": throughput
            })
            print(f"Batch {batch_size:3d}: {avg_time:6.3f}s | {throughput:6.2f} items/s")
    
    # Tests de concurrencia
    print("\n🔄 Tests de concurrencia:")
    print("-" * 60)
    
    for workers in [1, 2, 4]:
        start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(test_text_batch, 10) for _ in range(10)]
            results = [f.result() for f in futures]
        elapsed = time.time() - start
        
        total_items = sum(r[1] for r in results if r[0])
        throughput = total_items / elapsed
        print(f"Workers {workers}: {elapsed:6.3f}s | {throughput:6.2f} items/s")
    
    print("\n" + "=" * 60)
    print("✅ Benchmark completado!")

if __name__ == "__main__":
    try:
        # Verificar que el servicio está disponible
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            benchmark()
        else:
            print("❌ Servicio no disponible")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servicio. Verifica que esté corriendo.")
        sys.exit(1)
