#!/usr/bin/env bash
set -e

# Default CUDA tag if not provided
CUDA_TAG="${1:-cu126}"

echo "Installing PyTorch + torchvision for ${CUDA_TAG}..."
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "Installing Detectron2..."
pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo "Installing remaining requirements..."
pip install -r requirements.txt

echo "Installing xtcocotools..."
pip install --no-build-isolation --no-deps xtcocotools

echo ""
echo "Install complete."
