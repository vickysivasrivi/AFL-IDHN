#!/bin/sh

# Exit immediately if any command fails
set -e

if [ "$ROLE" = "server" ]; then
    echo "--- Starting in SERVER mode ---"
    exec python src/server_federated_mqtt.py

elif [ "$ROLE" = "client" ]; then
    # Check if CLIENT_ID is provided by the launcher script
    if [ -z "$CLIENT_ID" ]; then
        echo "Error: ROLE is 'client' but CLIENT_ID is not set."
        exit 1
    fi
    
    echo "--- Starting in CLIENT mode for Client ID: $CLIENT_ID ---"
    
    # Dynamically set the data file path for the Python script
    export DATA_FILE_PATH="data/federated_data/client_${CLIENT_ID}_data.csv"
    
    echo "Data file path set to: $DATA_FILE_PATH"
    
    # 'exec' replaces the shell process with the python process
    exec python src/client_federated_mqtt.py

else
    echo "Error: ROLE environment variable not set or invalid."
    echo "Please set ROLE to 'server' or 'client'."
    exit 1
fi