# Dockerfile optimizado para Jetson Orin Nano con Ubuntu 22.04 y Python 3.10

# Usar imagen base L4T sin PyTorch (más estable y confiable)
# R36.4 (JetPack 6.x)
FROM nvcr.io/nvidia/l4t-base:r36.4.0

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements primero (cache layer)
COPY ./app/requirements.txt /app/requirements.txt

# Instalar dependencias del sistema y Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar PyTorch para Jetson (desde NVIDIA)
RUN pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Instalar dependencias de la aplicación
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY ./app /app

# Crear directorio para caché de modelos
RUN mkdir -p /app/.cache/huggingface

# Pre-descargar el modelo CLIP (opcional, para builds más rápidas)
# RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Exponer el puerto de FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

# Comando optimizado para producción con workers y configuración para GPU
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]