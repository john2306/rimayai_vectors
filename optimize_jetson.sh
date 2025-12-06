#!/bin/bash
# Script de optimización para Jetson Orin Nano

echo "🚀 Optimizando Jetson Orin Nano para máximo rendimiento..."

# 1. Configurar modo de rendimiento máximo
echo "⚙️ Configurando modo de rendimiento máximo..."
sudo nvpmodel -m 0  # Modo máximo rendimiento (25W)
sudo jetson_clocks  # Fijar clocks al máximo

# 2. Verificar estado
echo "📊 Estado del sistema:"
sudo nvpmodel -q
tegrastats --interval 2000 &
TEGRA_PID=$!
sleep 3
kill $TEGRA_PID

# 3. Configurar variables de entorno para PyTorch
echo "🔧 Configurando variables de entorno..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# 4. Limpiar caché
echo "🧹 Limpiando caché..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo "✅ Optimización completada!"
echo "💡 Ejecuta: source optimize_jetson.sh (con sudo si es necesario)"
