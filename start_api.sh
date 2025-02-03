#!/bin/bash

# Define paths
VENV_PATH="/home/ubuntu/blockchainForecastAI"
SCRIPT_PATH="/home/ubuntu/blockchainForecastAI/api.py"
LOG_FILE="/home/ubuntu/blockchainForecastAI/output.log"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the script in the background using nohup
nohup python3 "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

# Print process ID
echo "Script started with PID: $!"
