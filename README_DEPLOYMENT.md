# Guía de Deployment en Jetson Orin Nano

## 🚀 Deployment Rápido

### Prerequisitos
- Jetson Orin Nano con JetPack 6.x (Ubuntu 22.04)
- Docker instalado
- NVIDIA Container Runtime

### Instalación de Dependencias

```bash
# Instalar Docker (si no está instalado)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Instalar NVIDIA Container Runtime
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verificar instalación
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r36.4.0 nvidia-smi
```

### Deployment

```bash
# 1. Dar permisos de ejecución a los scripts
chmod +x deploy.sh optimize_jetson.sh

# 2. Optimizar el sistema (opcional pero recomendado)
sudo ./optimize_jetson.sh

# 3. Desplegar la aplicación
./deploy.sh
```

### Uso Manual con Docker Compose

```bash
# Build y start
docker compose up -d --build

# Ver logs
docker compose logs -f

# Reiniciar
docker compose restart

# Detener
docker compose down

# Rebuild completo
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 📊 Benchmark

```bash
# Instalar dependencias
pip install requests numpy

# Ejecutar benchmark
python benchmark.py
```

## ⚙️ Optimizaciones Implementadas

### 1. **FP16 (Half Precision)**
- Reduce uso de memoria en ~50%
- Aumenta velocidad de inferencia en ~2x
- Automático en GPU, FP32 en CPU

### 2. **torch.compile()**
- Compilación JIT del modelo
- Reduce overhead de PyTorch
- ~20-30% más rápido

### 3. **CUDA Optimizations**
- `cudnn.benchmark=True`: Auto-optimiza convolutions
- `allow_tf32=True`: Usa TensorFloat32 para matmul
- Cache allocation optimizado

### 4. **Normalización de Embeddings**
- Vectores normalizados listos para similaridad coseno
- No requiere post-procesamiento

### 5. **Gestión de Memoria**
- Limpieza automática de CUDA cache
- Garbage collection estratégico
- Límites de memoria en Docker

### 6. **Uvloop**
- Event loop más rápido que asyncio
- ~2x más rápido en I/O

## 🔧 Configuración de Rendimiento

### Máximo Rendimiento (25W)
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Modo Balanceado (15W)
```bash
sudo nvpmodel -m 1
```

### Verificar Configuración
```bash
sudo nvpmodel -q
tegrastats
```

## 📈 Monitoreo

### Estadísticas en Tiempo Real
```bash
# Stats del sistema Jetson
tegrastats

# Stats de Docker
docker stats

# Logs de la aplicación
docker-compose logs -f
```

### Endpoints Disponibles

- **API Docs**: http://localhost:8000/docs
- **Vectorizar Batch**: POST http://localhost:8000/vectorize/batch

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```bash
# Reducir batch size o usar más limpieza de memoria
# Editar main.py y ajustar PYTORCH_CUDA_ALLOC_CONF
```

### Error: "nvidia-runtime not found"
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Servicio lento
```bash
# Verificar modo de rendimiento
sudo nvpmodel -q

# Configurar máximo rendimiento
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 📦 Estructura del Proyecto

```
rimayai_vectors2/
├── app/
│   ├── main.py              # API optimizada con FP16 y torch.compile
│   └── requirements.txt     # Dependencias
├── Dockerfile               # Imagen optimizada para Jetson
├── docker-compose.yml       # Configuración de deployment
├── deploy.sh               # Script de deployment automático
├── optimize_jetson.sh      # Script de optimización del sistema
├── benchmark.py            # Script de benchmark
└── README_DEPLOYMENT.md    # Esta guía
```

## 🎯 Rendimiento Esperado

En Jetson Orin Nano (8GB):
- **Texto**: ~50-100 embeddings/segundo (batch 10)
- **Imágenes**: ~20-40 embeddings/segundo (batch 10)
- **Latencia**: ~100-200ms (batch pequeño)
- **Memoria GPU**: ~2-3GB con modelo cargado

## 🔐 Producción

Para producción, considera:

1. **HTTPS**: Usar nginx como reverse proxy
2. **Autenticación**: Agregar API keys
3. **Rate Limiting**: Limitar requests por IP
4. **Logging**: Configurar logs estructurados
5. **Monitoring**: Prometheus + Grafana

## 📝 Notas

- El modelo CLIP se descarga en el primer uso (~600MB)
- Se cachea en `/app/.cache/huggingface` (persistente)
- Los embeddings están normalizados (norma L2 = 1)
- Compatible con similaridad coseno directa
