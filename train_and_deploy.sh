#!/bin/bash

# --- Environment Setup ---
# This section ensures the script uses the correct Conda environment.

# IMPORTANT: Replace this path with the correct path to your conda.sh
# Note the forward slashes and how C:\ is replaced with /c/ for Git Bash
CONDA_PATH="/c/Users/vicky/Miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "ERROR: conda.sh not found at '$CONDA_PATH'. Please update the path."
    exit 1
fi

echo "Activating Conda environment 'thesis'..."
conda activate thesis

# --- Configuration (remains the same) ---
BROKER_HOST="192.168.0.250"
SERVER_IP="192.168.0.250"
HTTP_PORT="8000"
MODEL_FILENAME="final_global_model_adaptive.weights.h5"
MODEL_FILE_PATH="saved_models/$MODEL_FILENAME"
MQTT_TOPIC_NOTIFICATION="afl/global/model_ready"

# --- Main Script ---

# Function to clean up background processes on exit
cleanup() {
    echo "--- Cleaning up background processes ---"
    if [ -n "$HTTP_PID" ]; then
        kill "$HTTP_PID"
    fi
    exit
}

# Trap script exit signals to run the cleanup function
trap cleanup EXIT SIGINT SIGTERM

echo "================================================="
echo " STEP 1: Starting HTTP file server in the background..."
echo "================================================="
python -m http.server $HTTP_PORT --bind 0.0.0.0 &
HTTP_PID=$!
echo "HTTP Server started with PID $HTTP_PID. Serving files on http://${SERVER_IP}:${HTTP_PORT}"

echo "================================================="
echo " STEP 2: Starting the Federated Learning Server..."
echo "================================================="

# IMPROVEMENT: Add 'set -e' to make the script exit immediately if a command fails
set -e

# Now this python command will use the activated 'thesis' environment
python src/server_federated_mqtt.py

# 'set +e' returns to the default behavior (don't exit on error)
set +e

echo "================================================="
echo " STEP 3: Training complete. Checking for model file..."
echo "================================================="

if [ -f "$MODEL_FILE_PATH" ]; then
    echo "Model file found at '$MODEL_FILE_PATH'."
    DOWNLOAD_URL="http://${SERVER_IP}:${HTTP_PORT}/${MODEL_FILE_PATH}"
    
    echo "STEP 4: Publishing notification to MQTT topic '$MQTT_TOPIC_NOTIFICATION'"
    echo "Download URL: $DOWNLOAD_URL"
    
    # This command will now be found in the 'thesis' environment
    paho_mqtt pub --host "$BROKER_HOST" -t "$MQTT_TOPIC_NOTIFICATION" -m "$DOWNLOAD_URL"
    
    echo "SUCCESS: Notification published."
else
    echo "ERROR: Model file was not found after training. Notification aborted."
fi

# The cleanup function will be called automatically when the script exits