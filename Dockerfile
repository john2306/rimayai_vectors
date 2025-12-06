# Dockerfile

# Usar una imagen base de NVIDIA optimizada para Jetson (L4T) con PyTorch y CUDA
# Asegúrate de que esta imagen base coincida con tu versión de JetPack/CUDA (ej. 36.x y CUDA 12.x)
# Usaremos una imagen base general que debería funcionar con CUDA 12.x
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación
COPY ./app /app

# Instalar las dependencias de Python
# Nota: La imagen base de NVIDIA ya tiene torch con CUDA. 
# Solo necesitamos instalar las otras dependencias.
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para correr la aplicación con Uvicorn
# La opción --host 0.0.0.0 es necesaria para que Docker sea accesible externamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]