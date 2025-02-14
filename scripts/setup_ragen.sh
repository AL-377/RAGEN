#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is available"
        return 0
    else
        echo "Conda is not installed. Please install Conda first."
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_conda || exit 1
    
    # Create and activate conda environment
    print_step "Creating conda environment 'ragen' with Python 3.9..."
    conda create -n ragen python=3.9 -y
    
    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    conda activate ragen
    
    # Clone repository
    print_step "Cloning ragen repository..."
    # git clone git@github.com:ZihanWang314/ragen.git
    # cd ragen
    
    # Install package in editable mode
    print_step "Installing ragen package..."
    pip install -e .
    
    # Install PyTorch with CUDA if available
    if check_cuda; then
        print_step "Installing PyTorch with CUDA support..."
        pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
        
        # Install flash-attention if CUDA is available
        print_step "Installing CUDA toolkit and flash-attention..."
        conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
        export CUDA_HOME=$CONDA_PREFIX
        pip3 install flash-attn --no-build-isolation
    else
        print_step "Installing PyTorch without CUDA support..."
        pip install torch==2.4.0
    fi
    
    # Install remaining requirements
    print_step "Installing additional requirements..."
    pip install -r requirements.txt
    
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "To activate the environment, run: conda activate ragen"
}

# Run main installation
main
