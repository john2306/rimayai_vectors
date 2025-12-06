# USAR ESTA IMAGEN: Es la única que existe para r36.4.0 y trae CUDA incluido
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    # Importante para que Python encuentre las librerías de CUDA internas
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Instalar dependencias básicas
# Nota: l4t-jetpack ya es pesada y completa, instalamos solo lo que falte
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

# INSTALACIÓN CORRECTA DE PYTORCH
# Para JetPack 6.1 (R36.4), usamos la versión compatible compilada por NVIDIA.
# La URL abajo está verificada para JP 6.1
RUN pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Copiar requirements (Asegúrate de quitar 'torch' de este archivo)
COPY ./app/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código
COPY ./app /app
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]