#!/bin/bash
# Automated setup script for JailbreakingLLMs UV virtual environment
# This script creates a UV venv and installs all dependencies automatically

set -e  # Exit on any error

echo "=========================================="
echo "JailbreakingLLMs UV Environment Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  or: pip install uv"
    exit 1
fi

echo "✓ UV is installed"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Check if venv already exists
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists at .venv"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "ℹ️  Using existing virtual environment"
        echo ""
        echo "To activate the environment, run:"
        echo "  source .venv/bin/activate"
        exit 0
    fi
fi

echo "🔨 Creating UV virtual environment..."
uv venv

echo ""
echo "✅ Virtual environment created"
echo ""

# Activate the virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "📦 Step 1/4: Installing PyTorch with CUDA 12.1 support..."
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📦 Step 2/4: Installing fschat (bypassing dependency conflicts)..."
uv pip install fschat==0.2.23 --no-deps

echo ""
echo "📦 Step 3/4: Installing other dependencies..."
uv pip install pandas seaborn matplotlib numpy psutil \
    litellm==1.30.0 datasets evaluate anthropic google-generativeai \
    openai wandb jailbreakbench transformers>=4.35.0 accelerate peft

echo ""
echo "📦 Step 4/4: Installing vllm (separately to avoid building from source)..."
uv pip install "vllm>=0.6.0"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Set your API keys:"
echo "     export OPENAI_API_KEY=your_key_here"
echo "     export ANTHROPIC_API_KEY=your_key_here"
echo "     export GOOGLE_API_KEY=your_key_here"
echo ""
echo "  2. Login to Weights & Biases (optional):"
echo "     wandb login"
echo ""
echo "  3. Run experiments:"
echo "     python3 main.py --attack-model vicuna --target-model vicuna --judge-model gpt-4 --goal \"...\" --target-str \"...\""
echo ""
