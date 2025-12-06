# -----------------------------------------------------------------------
# CORRECCIÓN PRINCIPAL: Usamos r36.3.0 que sí existe y es compatible con JP6
# -----------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-base:r36.3.0

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    # Asegura que Python encuentre las librerías CUDA
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (He añadido libopenmpi-dev que suele pedir PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libopenmpi-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------
# INSTALACIÓN DE PYTORCH
# Nota: Si este enlace falla (404), cambia a la versión 2.3.0 que es la estándar de JP6
# URL Alternativa estable: https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0+nv24.03-cp310-cp310-linux_aarch64.whl
# -----------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Copiar requirements primero (cache layer)
COPY ./app/requirements.txt /app/requirements.txt

# Instalar dependencias.
# IMPORTANTE: Asegúrate de que en requirements.txt NO esté 'torch' otra vez
# para evitar conflictos. 'torchvision' debe instalarse sin dependencias de torch.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY ./app /app

# Crear directorio para caché de modelos y asignar permisos si es necesario
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Exponer el puerto de FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

# Comando optimizado
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]