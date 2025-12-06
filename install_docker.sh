#!/bin/bash
# Script de instalación de Docker y Docker Compose para Jetson Orin Nano
# Ubuntu 22.04

set -e

echo "🚀 Instalación de Docker y Docker Compose en Jetson Orin Nano"
echo "=============================================================="

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Actualizar sistema
echo -e "${YELLOW}📦 Actualizando sistema...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# 2. Instalar dependencias
echo -e "${YELLOW}📦 Instalando dependencias...${NC}"
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git

# 3. Verificar si Docker ya está instalado
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker ya está instalado${NC}"
    docker --version
else
    echo -e "${YELLOW}🐳 Instalando Docker...${NC}"
    
    # Para Jetson, Docker generalmente viene preinstalado con JetPack
    # Si no, instalarlo manualmente
    sudo apt-get install -y docker.io
    
    # Agregar usuario al grupo docker
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✅ Docker instalado${NC}"
fi

# 4. Habilitar y arrancar Docker
echo -e "${YELLOW}⚙️ Configurando Docker...${NC}"
sudo systemctl enable docker
sudo systemctl start docker

# 5. Instalar NVIDIA Container Runtime (crítico para Jetson)
echo -e "${YELLOW}🎮 Instalando NVIDIA Container Runtime...${NC}"

if ! dpkg -l | grep -q nvidia-container-runtime; then
    # Agregar repositorio
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-runtime nvidia-docker2
    
    # Reiniciar Docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✅ NVIDIA Container Runtime instalado${NC}"
else
    echo -e "${GREEN}✅ NVIDIA Container Runtime ya está instalado${NC}"
fi

# 6. Instalar Docker Compose (v2)
echo -e "${YELLOW}📦 Instalando Docker Compose...${NC}"

# Verificar si ya está instalado
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Compose ya está instalado${NC}"
    docker compose version || docker-compose --version
else
    # Método 1: Instalar docker-compose-plugin (recomendado)
    sudo apt-get install -y docker-compose-plugin
    
    # Si falla, usar método manual
    if ! docker compose version &> /dev/null 2>&1; then
        echo -e "${YELLOW}📥 Instalando Docker Compose manualmente...${NC}"
        
        # Detectar arquitectura
        ARCH=$(dpkg --print-architecture)
        
        # Descargar última versión
        DOCKER_COMPOSE_VERSION="v2.24.0"
        sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-${ARCH}" \
            -o /usr/local/bin/docker-compose
        
        sudo chmod +x /usr/local/bin/docker-compose
        
        # Crear symlink
        sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    fi
    
    echo -e "${GREEN}✅ Docker Compose instalado${NC}"
fi

# 7. Verificar instalación
echo ""
echo -e "${YELLOW}🔍 Verificando instalación...${NC}"
echo "----------------------------------------"

echo "Docker version:"
docker --version

echo ""
echo "Docker Compose version:"
docker compose version 2>/dev/null || docker-compose --version

echo ""
echo "NVIDIA Container Runtime:"
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r36.4.0 nvidia-smi || \
    echo "⚠️ No se pudo verificar nvidia-runtime (puede requerir reinicio)"

# 8. Configurar permisos (requiere re-login)
echo ""
echo -e "${YELLOW}⚙️ Configurando permisos...${NC}"
sudo groupadd -f docker
sudo usermod -aG docker $USER

# 9. Resumen final
echo ""
echo -e "${GREEN}=============================================================="
echo "✅ Instalación completada!"
echo "==============================================================${NC}"
echo ""
echo "📋 Próximos pasos:"
echo ""
echo "1️⃣ Cerrar sesión y volver a iniciar para aplicar permisos:"
echo "   logout  # o reiniciar: sudo reboot"
echo ""
echo "2️⃣ Verificar que funciona sin sudo:"
echo "   docker ps"
echo "   docker compose version"
echo ""
echo "3️⃣ Desplegar tu aplicación:"
echo "   cd rimayai_vectors2"
echo "   docker compose up -d --build"
echo ""
echo "⚠️ IMPORTANTE: Debes cerrar sesión o reiniciar para que los"
echo "   cambios de grupo (docker) tomen efecto."
echo ""
