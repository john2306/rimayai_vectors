#!/bin/bash
# Script de deployment para Jetson Orin Nano

set -e

echo "🚀 Deployment de Vectorizador CLIP en Jetson Orin Nano"
echo "=========================================================="

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Verificar que estamos en Jetson
echo -e "${YELLOW}📋 Verificando hardware...${NC}"
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    echo "⚠️ No se detectó Jetson. Continuando de todas formas..."
fi

# 2. Optimizar sistema
echo -e "${YELLOW}⚙️ Optimizando sistema...${NC}"
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0  # Modo máximo rendimiento
    sudo jetson_clocks   # Máximo clock
    echo -e "${GREEN}✅ Sistema optimizado para máximo rendimiento${NC}"
else
    echo "⚠️ nvpmodel no encontrado, saltando optimización de Jetson"
fi

# 3. Verificar Docker y nvidia-runtime
echo -e "${YELLOW}🐳 Verificando Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no instalado. Instala Docker primero."
    exit 1
fi

if ! docker info | grep -q nvidia; then
    echo "⚠️ Advertencia: nvidia-runtime no detectado"
    echo "Instala con: sudo apt-get install nvidia-docker2"
fi

# 4. Detener contenedores previos
echo -e "${YELLOW}🛑 Deteniendo contenedores previos...${NC}"
docker compose down 2>/dev/null || true

# 5. Limpiar recursos
echo -e "${YELLOW}🧹 Limpiando recursos...${NC}"
docker system prune -f
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 6. Build de la imagen
echo -e "${YELLOW}🏗️ Construyendo imagen Docker...${NC}"
docker compose build --no-cache

# 7. Iniciar servicio
echo -e "${YELLOW}🚀 Iniciando servicio...${NC}"
docker compose up -d

# 8. Esperar a que el servicio esté listo
echo -e "${YELLOW}⏳ Esperando a que el servicio esté listo...${NC}"
sleep 10

# 9. Verificar estado
echo -e "${YELLOW}📊 Verificando estado del servicio...${NC}"
docker compose ps
docker compose logs --tail=50

# 10. Test de salud
echo -e "${YELLOW}🏥 Test de salud...${NC}"
if curl -f http://localhost:8000/docs &> /dev/null; then
    echo -e "${GREEN}✅ Servicio funcionando correctamente!${NC}"
    echo -e "${GREEN}📖 Documentación: http://$(hostname -I | awk '{print $1}'):8000/docs${NC}"
else
    echo "❌ Servicio no responde. Revisa los logs:"
    docker compose logs
fi

# 11. Mostrar stats
echo -e "${YELLOW}📈 Estadísticas del sistema:${NC}"
tegrastats --interval 1000 &
STATS_PID=$!
sleep 3
kill $STATS_PID

echo ""
echo -e "${GREEN}=========================================================="
echo "🎉 Deployment completado!"
echo "=========================================================="
echo ""
echo "Comandos útiles:"
echo "  Ver logs:     docker compose logs -f"
echo "  Reiniciar:    docker compose restart"
echo "  Detener:      docker compose down"
echo "  Stats GPU:    tegrastats"
echo -e "${NC}"
