#!/bin/bash
# Ensure the script exits if a command fails
set -e

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate my-docker-environment

# Run the main application
exec python -m detect_malignant --config batch/config.json
