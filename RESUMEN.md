# 🚀 Vectorizador CLIP Optimizado para Jetson Orin Nano

## ✨ Optimizaciones Implementadas

### 🔥 Rendimiento
- ✅ **FP16 (Half Precision)**: 50% menos memoria, 2x más rápido
- ✅ **torch.compile()**: JIT compilation para ~30% mejora
- ✅ **CUDA Optimizations**: cuDNN benchmark, TF32 matmul
- ✅ **Embeddings normalizados**: Listos para similaridad coseno
- ✅ **Uvloop**: Event loop 2x más rápido
- ✅ **Gestión de memoria**: Auto-limpieza de CUDA cache

### 📦 Deployment
- ✅ **Docker optimizado**: Imagen L4T específica para Jetson
- ✅ **Docker Compose**: Deployment con un comando
- ✅ **Health checks**: Monitoreo automático
- ✅ **Scripts automatizados**: deploy.sh y optimize_jetson.sh

### 📊 Monitoreo
- ✅ **Endpoints de salud**: /health, /metrics
- ✅ **Logs estructurados**: Información detallada
- ✅ **Benchmark script**: Evaluación de rendimiento

## 📂 Archivos Creados/Modificados

```
✅ app/main.py              - API optimizada con FP16 y torch.compile
✅ app/requirements.txt     - Dependencias actualizadas
✅ Dockerfile               - Imagen optimizada para Jetson L4T
✅ docker-compose.yml       - Configuración de deployment
✅ deploy.sh               - Script de deployment automático
✅ optimize_jetson.sh      - Script de optimización del sistema
✅ benchmark.py            - Script de evaluación de rendimiento
✅ examples.py             - Ejemplos de uso del API
✅ README_DEPLOYMENT.md    - Guía completa de deployment
✅ .dockerignore           - Optimización del build
```

## 🎯 Quick Start en Jetson Orin Nano

### 1. Preparar el Sistema
```bash
# Dar permisos
chmod +x deploy.sh optimize_jetson.sh

# Optimizar sistema (opcional pero recomendado)
sudo ./optimize_jetson.sh
```

### 2. Desplegar
```bash
# Opción A: Deployment automático
./deploy.sh

# Opción B: Manual con docker-compose
docker compose up -d --build
```

### 3. Verificar
```bash
# Ver logs
docker compose logs -f

# Health check
curl http://localhost:8000/health

# Métricas
curl http://localhost:8000/metrics

# Docs interactivos
# Abrir en navegador: http://<IP_JETSON>:8000/docs
```

### 4. Probar
```bash
# Instalar dependencias para ejemplos
pip install requests numpy

# Ejecutar ejemplos
python examples.py

# Benchmark
python benchmark.py
```

## 📊 Rendimiento Esperado

En Jetson Orin Nano (8GB, modo 25W):
- **Texto**: ~50-100 embeddings/seg (batch 10)
- **Imágenes**: ~20-40 embeddings/seg (batch 10)
- **Latencia**: ~100-200ms (batch pequeño)
- **Memoria GPU**: ~2-3GB con modelo cargado

## 🔧 Configuración de Rendimiento

```bash
# Máximo rendimiento (25W)
sudo nvpmodel -m 0
sudo jetson_clocks

# Verificar
sudo nvpmodel -q
tegrastats
```

## 📡 Endpoints Disponibles

- `GET  /`                  - Información del servicio
- `GET  /health`            - Health check
- `GET  /metrics`           - Métricas del sistema
- `GET  /docs`              - Documentación interactiva
- `POST /vectorize/batch`   - Vectorizar texto/imágenes

## 🧪 Ejemplo de Uso

```python
import requests

# Vectorizar textos
payload = {
    "items": [
        {"text": "un gato negro"},
        {"text": "un perro grande"}
    ]
}

response = requests.post(
    "http://localhost:8000/vectorize/batch",
    json=payload
)

data = response.json()
print(f"Vectores: {len(data['embeddings'])}")
print(f"Dimensión: {len(data['embeddings'][0]['vector'])}")
```

## 🐛 Troubleshooting

### CUDA out of memory
```bash
# Editar docker-compose.yml y reducir límite de memoria
# O reducir batch size en las requests
```

### Servicio lento
```bash
# Verificar modo de rendimiento
sudo nvpmodel -q

# Activar máximo rendimiento
sudo nvpmodel -m 0
sudo jetson_clocks
```

### nvidia-runtime not found
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 📈 Monitoreo en Producción

```bash
# Stats del sistema
tegrastats

# Stats de Docker
docker stats

# Logs
docker-compose logs -f

# Métricas via API
curl http://localhost:8000/metrics | jq
```

## 🔐 Mejoras para Producción

Considera implementar:
1. HTTPS con nginx reverse proxy
2. Autenticación con API keys
3. Rate limiting
4. Logging estructurado (JSON)
5. Monitoring con Prometheus + Grafana

## 📝 Notas Importantes

- El modelo CLIP (~600MB) se descarga en el primer uso
- Los embeddings están normalizados (L2 norm = 1)
- Compatible con similaridad coseno directa
- Cache persistente en volumen Docker

## 🎉 ¡Listo para Producción!

Tu vectorizador CLIP está optimizado y listo para correr en Jetson Orin Nano con:
- ⚡ Máximo rendimiento (FP16, torch.compile)
- 🔒 Containerizado con Docker
- 📊 Monitoreo completo
- 🚀 Deployment automatizado

---

**Desarrollado para Jetson Orin Nano | Ubuntu 22.04 | Python 3.10**
