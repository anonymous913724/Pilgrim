#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the name of the virtual environment directory
ENV_NAME="venv"

echo "Creating a virtual environment..."
# Create the virtual environment using the venv module
python3 -m venv $ENV_NAME

echo "Activating the virtual environment..."
# Activate the environment (source command is shell-specific)
source $ENV_NAME/bin/activate

# we need to install torch geometric separately because it has specific installation instructions
echo "Installing torch geometric and its dependencies (we need to do this first)"
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

echo "Installing dependencies from requirements.txt..."
# Install Python packages using pip within the activated environment
pip install -r requirements.txt

echo "Setup complete. To use the environment, run 'source $ENV_NAME/bin/activate'"