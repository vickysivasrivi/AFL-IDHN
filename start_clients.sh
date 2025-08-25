#!/bin/bash

# This script is a scalable launcher for the federated learning clients.
# It works with the generic 'client' service defined in docker-compose.yml.
#
# Usage: ./start_clients.sh <number_of_clients>
# Example: ./start_clients.sh 5  (This will start 5 clients, from ID 0 to 4)

# --- Check for input ---
if [ -z "$1" ]; then
    echo "Error: Please provide the number of clients to start."
    echo "Usage: $0 <number_of_clients>"
    exit 1
fi

NUM_CLIENTS=$1
echo "--- Starting $NUM_CLIENTS client containers in the background... ---"

# --- Main Loop ---
# Loop from 0 to (NUM_CLIENTS - 1)
for i in $(seq 0 $(($NUM_CLIENTS - 1))); do
    echo "Launching client with ID: ${i}"
    
    # Use 'docker-compose run' to start a new instance of the 'client' service.
    # -d: Run in detached (background) mode.
    # --rm: Automatically remove the container when it stops.
    # -e CLIENT_ID=$i: Our entrypoint.sh script will use this.
    # --name "client_${i}": Give each container a unique, predictable name.
    docker-compose run -d --rm \
        -e CLIENT_ID=$i \
        --name "client_${i}" \
        client
done

echo "--- All $NUM_CLIENTS client containers have been launched. ---"
echo "Use 'docker ps' to see them running or 'docker-compose logs -f' to view their output."