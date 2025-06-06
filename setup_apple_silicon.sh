#!/bin/bash
# Setup script for Robot Arm Simulation on Apple Silicon Mac

echo "🍎 Setting up Robot Arm Simulation for Apple Silicon Mac..."

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is designed for Apple Silicon Macs"
    echo "   Detected architecture: $(uname -m)"
    echo "   Continuing anyway..."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Homebrew if not present
if ! command_exists brew; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "✅ Homebrew already installed"
fi

# Install Miniconda if not present
if ! command_exists conda; then
    echo "🐍 Installing Miniconda for Apple Silicon..."
    brew install --cask miniconda
    
    # Initialize conda
    ~/miniconda3/bin/conda init zsh
    source ~/.zshrc
else
    echo "✅ Conda already installed"
fi

# Install UV if not present
if ! command_exists uv; then
    echo "⚡ Installing UV (fast Python package manager)..."
    brew install uv
else
    echo "✅ UV already installed"
fi

echo "🤖 Creating PyBullet environment for Apple Silicon..."

# Create conda environment with Python 3.11
conda create -n pybullet-env python=3.11 -y

# Activate environment
echo "🔧 Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pybullet-env

# Install PyBullet via conda-forge (better Apple Silicon support)
echo "🔩 Installing PyBullet via conda-forge..."
conda install -c conda-forge pybullet -y

# Install core scientific packages via conda (better optimized for Apple Silicon)
echo "📊 Installing core scientific packages..."
conda install -c conda-forge numpy scipy matplotlib -y

# Install remaining packages with UV (faster)
echo "📦 Installing remaining packages with UV..."
uv pip install pyyaml torch transformers stable-baselines3 gymnasium opencv-python Pillow tqdm

# Optional: Try to install OpenGL packages (may fail on Apple Silicon)
echo "🎮 Attempting to install OpenGL packages (may fail on Apple Silicon)..."
uv pip install PyOpenGL PyOpenGL-accelerate glfw moderngl || echo "⚠️  OpenGL packages failed to install (expected on Apple Silicon)"

echo "🧪 Testing installation..."

# Test basic imports
python -c "
import sys
print(f'Python: {sys.version}')
print(f'Platform: {sys.platform}')

try:
    import pybullet
    print('✅ PyBullet imported successfully')
except ImportError as e:
    print(f'❌ PyBullet import failed: {e}')

try:
    import numpy as np
    print('✅ NumPy imported successfully')
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import torch
    print('✅ PyTorch imported successfully')
    if torch.backends.mps.is_available():
        print('🚀 Metal Performance Shaders (MPS) available for GPU acceleration!')
    else:
        print('💻 Using CPU (MPS not available)')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import transformers
    print('✅ Transformers imported successfully')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your terminal or VSCode"
echo "2. Activate the environment: conda activate pybullet-env"
echo "3. Navigate to project: cd robot-simulation"
echo "4. Test basic functionality: python test_basic.py"
echo "5. Run robot simulation: python main.py --demo"
echo ""
echo "💡 Apple Silicon Notes:"
echo "- 3D visualization is disabled (OpenGL compatibility issues)"
echo "- GUI is disabled (Tkinter compatibility issues)"
echo "- Use command line interface: python main.py --command 'wave hello'"
echo "- Physics simulation works perfectly with PyBullet"
echo "- ML training works with MPS acceleration if available"
echo ""
echo "🚀 Ready to simulate robots!"
