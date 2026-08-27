# Pipeline 3D de Residuos Fluviales 🌊🤖

Pipeline automatizado de visión por computadora y reconstrucción 3D para la detección, segmentación, estimación de profundidad métrica, inpainting de fondos, triangulación de terreno y análisis volumétrico de residuos sólidos en entornos fluviales.

---

## 🚀 Requisitos del Sistema
- **SO**: Linux (Ubuntu / WSL2 en Windows)
- **GPU**: NVIDIA con soporte CUDA (recomendado 8GB+ VRAM)
- **Gestor de paquetes**: `uv` (Astral)
- **Python**: 3.11+

---

## 📦 Habilitación de Repositorios y Compiladores del Sistema

Para evitar errores críticos de compilación, cabeceras faltantes de C++ o problemas al compilar dependencias pesadas de PyTorch, Open3D o Depth Pro en Linux/WSL, asegúrate de habilitar los repositorios oficiales de software y actualizar las herramientas de compilación esenciales antes de instalar el entorno:

```bash
# 1. Habilitar repositorios Universe/Multiverse (esenciales en Ubuntu/Debian para paquetes gráficos y multimedia)
sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo apt update

# 2. Instalar herramientas de compilación base (gcc, g++, make, cmake, build-essential)
sudo apt install -y build-essential cmake git curl wget pkg-config

# 3. Instalar librerías de soporte gráfico y de procesamiento de imágenes requeridas por Open3D y OpenCV
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

## ⚙️ Instalación y Configuración del Entorno

Una vez preparados los repositorios del sistema y las herramientas de compilación, configura las variables de entorno para conexiones lentas y sincroniza el entorno virtual con uv:
```Bash

# Evitar timeouts en descargas pesadas de paquetes de IA
export UV_HTTP_TIMEOUT=600

# Sincronizar dependencias del proyecto
uv sync
```

