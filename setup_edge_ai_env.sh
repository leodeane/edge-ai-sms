#!/bin/bash
# Setup script for Edge AI Development Environment
# Installs UV, bootstraps Python 3.12, and installs dependencies from requirements.txt

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Setting up Edge AI Development Environment with UV and Python 3.12..."

# Check if running with sudo/root permissions
if [ "$EUID" -eq 0 ]; then
  echo "⚠️ Warning: This script is running with sudo/root permissions."
  echo "It's recommended to run without sudo to avoid permission issues with Python packages."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
  echo "🔧 Installing UV package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add UV to PATH for this session
  export PATH="$HOME/.local/bin:$PATH"
  
  # Verify UV installation
  uv --version
  echo "✅ UV installed successfully!"
else
  echo "✅ UV already installed, version: $(uv --version)"
fi

# Create requirements.txt file
echo "📝 Creating requirements.txt file..."
cat > requirements.txt << 'EOF'
# YOLO and Computer Vision
ultralytics

# Core ONNX packages
onnx
onnxruntime
optimum

# Deep Learning Frameworks
torch
torchvision
tensorflow

# Computer Vision Libraries
opencv-python-headless
pillow
scikit-image

# NLP & Speech Processing
transformers
datasets
soundfile
librosa
sentencepiece

# Hugging Face Tools
huggingface_hub
accelerate

# Model Optimization Tools
neural-compressor

# Utilities & Data Processing
numpy
scipy
pandas
matplotlib
seaborn
tqdm

# Development Tools
jupyterlab
ipywidgets

# Web UI (for demos)
streamlit
gradio

# API Development
fastapi
uvicorn[standard]

# Benchmarking & Profiling
psutil
py-spy

# Development dependencies
pytest
black
flake8
mypy
pre-commit
EOF

# Temporarily hide system Python to force UV to download its own
echo "🔧 Creating virtual environment with UV-managed Python 3.12..."
TEMP_PATH="$PATH"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] || [[ "$(uname -s)" == MINGW* ]]; then
    # On Windows, temporarily remove common Python paths
    export PATH=$(echo "$PATH" | sed 's|/c/Python[^:]*:||g' | sed 's|C:\\Python[^;]*;||g')
fi

# Create virtual environment - UV should download Python if not found
uv venv .venv --python 3.12

# Restore PATH
export PATH="$TEMP_PATH"

# Install dependencies from requirements.txt
echo "📦 Installing dependencies from requirements.txt (this may take a while)..."
# First install pip in the UV-created venv
uv pip install --python .venv pip

# Now install all requirements
uv pip install --python .venv -r requirements.txt

# Verify key installations
echo "🔍 Verifying installations..."
uv run --python .venv python -c "
try:
    import ultralytics, onnx, onnxruntime, torch, tensorflow, fastapi
    print('✅ Key packages verified:')
    print(f'  - Ultralytics: {ultralytics.__version__}')
    print(f'  - ONNX: {onnx.__version__}')
    print(f'  - ONNX Runtime: {onnxruntime.__version__}')
    print(f'  - PyTorch: {torch.__version__}')
    print(f'  - TensorFlow: {tensorflow.__version__}')
    print(f'  - FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "🎉 Edge AI environment setup is complete!"
echo "📋 Summary:"
echo "  - UV package manager installed"
echo "  - Python 3.12 virtual environment created in .venv/"
echo "  - All dependencies installed from requirements.txt"
echo ""
echo "🔄 To use this environment:"
echo "  Option 1 (Recommended): uv run --python .venv python your_script.py"
echo "  Option 2 (Manual activation):"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] || [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == CYGWIN* ]]; then
    echo "    - Windows/Git Bash: source .venv/Scripts/activate"
else
    echo "    - Linux/Mac: source .venv/bin/activate"
fi
echo "    - Then: python your_script.py"
echo "    - Deactivate: deactivate"
echo ""
echo "🚀 Ready for Edge AI development!"