#!/bin/bash

# Default environment
ENV_NAME=${1:-"sokoban"}

export PYTHONHASHSEED=10000

# Get and execute the training command
python ragen/train.py "$ENV_NAME" "$@" | bash