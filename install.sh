#!/usr/bin/env bash
set -e

# Default CUDA tag if not provided
CUDA_TAG="${1:-cu126}"

# Default venv directory if not provided
VENV_DIR="${2:-venv}"

echo "Creating virtual environment at ${VENV_DIR} (if missing)..."
python3 -m venv "${VENV_DIR}"

# Use venv binaries directly
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

echo "Upgrading pip in venv..."
"${PYTHON_BIN}" -m pip install -q --upgrade pip

echo "Installing PyTorch + torchvision for ${CUDA_TAG}..."
"${PIP_BIN}" install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "Installing Detectron2..."
"${PIP_BIN}" install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo "Installing remaining requirements..."
"${PIP_BIN}" install -r requirements.txt

echo "Installing xtcocotools..."
"${PIP_BIN}" install --no-build-isolation --no-deps xtcocotools

echo ""
echo "Install complete."
echo "Torch check:"
"${PYTHON_BIN}" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda)"

echo ""
echo "To activate later:"
echo "source ${VENV_DIR}/bin/activate"
