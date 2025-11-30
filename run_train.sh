#!/bin/bash
# Helper script to run training with correct LD_LIBRARY_PATH for C++ extension

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Set LD_LIBRARY_PATH to include torch libraries
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Run training
"$VENV_DIR/bin/python" src/cublino_contra/train.py "$@"
