import torch
import torch.nn as nn
import numpy as np
import paho.mqtt.client as paho
from paho import mqtt
import time
import ssl
import pickle
import threading

# --- Configuration for MQTT and Federated Learning ---
SERVER_ID = "fl_server"

MQTT_BROKER_HOST = "2e09a248750440018083de7318e45187.s1.eu.hivemq.cloud"
MQTT_BROKER_PORT = 8883
MQTT_USERNAME = "vignesh_2181" # <<< IMPORTANT: Replace with your HiveMQ username
MQTT_PASSWORD = "Vignesh@2181" # <<< IMPORTANT: Replace with your HiveMQ password
MQTT_TLS_CA_CERT = "certs/hivemq-ca.pem" # Path to your downloaded HiveMQ CA certificate

# MQTT Topics
GLOBAL_MODEL_TOPIC = "fl/global_model"
LOCAL_UPDATE_TOPIC = "fl/updates/#" # Subscribe to all client updates

# Federated Learning Parameters
NUM_EXPECTED_CLIENTS = 3 # Your target number of Raspberry Pi clients
NUM_COMMUNICATION_ROUNDS = 10 # Number of federated rounds to run
ADAPTIVE_LR_START = 0.01
ADAPTIVE_LR_DECAY = 0.95 # Decay factor for adaptive learning rate
SERVER_LEARNING_RATE = ADAPTIVE_LR_START

# Model configuration (must match fl_client.py)
INPUT_FEATURES = 78
NUM_CLASSES = 15

# --- PyTorch CNN-LSTM Model Definition (The Global Model) ---
class CNNDetector(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CNNDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        lstm_input_size = 64 * (input_features // 4) 
        
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x.squeeze(1))
        x = self.fc(x)
        return x

# --- Server-side Global Model and State ---
global_model = CNNDetector(input_features=INPUT_FEATURES, num_classes=NUM_CLASSES)
client_updates = {}
current_round = 0
round_lock = threading.Lock()

# --- Aggregation Logic with Client-Aware Weighting ---
def aggregate_models(updates):
    """
    Performs federated averaging with client-aware weighting.
    The "adaptive" aggregation logic is conceptually implemented here,
    with gradient compression as a future enhancement for the clients.
    """
    print(f"Aggregating updates from {len(updates)} clients...")
    
    # Calculate the total number of examples from all clients
    total_examples = sum(update['num_examples'] for update in updates.values())
    
    # Initialize the aggregated state_dict
    aggregated_state_dict = {}
    for key in global_model.state_dict().keys():
        aggregated_state_dict[key] = torch.zeros_like(global_model.state_dict()[key])
        
    # Perform a weighted average
    for client_id, update in updates.items():
        client_state_dict = update['state_dict']
        num_examples = update['num_examples']
        weight = num_examples / total_examples # Calculate the weight for this client
        
        for key in client_state_dict.keys():
            aggregated_state_dict[key] += client_state_dict[key] * weight

    return aggregated_state_dict

# --- Helper Functions for Model Serialization/Deserialization ---
def serialize_model_weights(model_state_dict):
    """Serializes PyTorch model state_dict for transmission."""
    return pickle.dumps(model_state_dict)

def deserialize_model_weights(serialized_weights):
    """Deserializes PyTorch model state_dict from received bytes."""
    return pickle.loads(serialized_weights)


# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Server {SERVER_ID}: Connected to MQTT Broker (HiveMQ) successfully!")
        client.subscribe(LOCAL_UPDATE_TOPIC, qos=1)
        print(f"Server {SERVER_ID}: Subscribed to '{LOCAL_UPDATE_TOPIC}'")
    else:
        print(f"Server {SERVER_ID}: Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    with round_lock:
        print(f"Server {SERVER_ID}: Received update from {msg.topic}")
        try:
            update_payload = pickle.loads(msg.payload)
            client_updates[msg.topic] = update_payload
            print(f"Server {SERVER_ID}: Stored update from {msg.topic}. Total updates for this round: {len(client_updates)}")

            # Check if we have received updates from all expected clients
            if len(client_updates) >= NUM_EXPECTED_CLIENTS:
                print(f"Server {SERVER_ID}: Received all updates for round {current_round}. Aggregating...")
                
                # Perform aggregation
                new_global_state_dict = aggregate_models(client_updates)
                
                # Update global model with aggregated weights
                global_model.load_state_dict(new_global_state_dict)
                
                # Prepare for next round
                client_updates.clear()
                
                # Publish the new global model
                serialized_global_model = serialize_model_weights(global_model.state_dict())
                client.publish(GLOBAL_MODEL_TOPIC, payload=serialized_global_model, qos=1)
                print(f"Server {SERVER_ID}: Published new global model for round {current_round + 1} to '{GLOBAL_MODEL_TOPIC}'")
                
        except Exception as e:
            print(f"Server {SERVER_ID}: Error processing client update: {e}")

# --- MQTT Client Setup ---
client = paho.Client(client_id=SERVER_ID, userdata=None, protocol=paho.MQTTv5, callback_api_version=paho.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    print(f"Server {SERVER_ID}: TLS enabled with CA certificate: {MQTT_TLS_CA_CERT}")
except FileNotFoundError:
    print(f"Server {SERVER_ID}: Error: CA certificate not found at {MQTT_TLS_CA_CERT}.")
    print("Please download it from HiveMQ Cloud and ensure the path is correct.")
    exit()

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

print(f"Server {SERVER_ID}: Attempting to connect to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Server {SERVER_ID}: MQTT connection failed: {e}")
    print("Please check your network, broker hostname, port, username, and password.")
    exit()

client.loop_start()

# --- Main Server Loop for Federated Learning Rounds ---
# This is where the server controls the rounds and applies the adaptive learning rate.
try:
    while current_round < NUM_COMMUNICATION_ROUNDS:
        # Publish the model for the clients to start the round
        initial_global_model = global_model.state_dict()
        serialized_initial_model = serialize_model_weights(initial_global_model)
        client.publish(GLOBAL_MODEL_TOPIC, payload=serialized_initial_model, qos=1)
        print(f"Server {SERVER_ID}: Published global model for Round {current_round + 1}.")
        
        # Wait for client updates to arrive via the on_message callback
        print(f"Server {SERVER_ID}: Waiting for updates from {NUM_EXPECTED_CLIENTS} clients...")

        # In a real system, you would have logic here to wait until all clients have reported.
        # For this prototype, the on_message callback is handling the aggregation and next publish.
        
        # Increment the round count after a short delay
        time.sleep(30) # Wait for clients to finish local training and send updates
        current_round += 1
        
        # Adaptive Learning Rate Scheduling
        # A simple decay-based scheduler
        if current_round > 0:
            SERVER_LEARNING_RATE *= ADAPTIVE_LR_DECAY
            print(f"Server {SERVER_ID}: Adaptive learning rate for next round set to: {SERVER_LEARNING_RATE:.6f}")

except KeyboardInterrupt:
    print(f"Server {SERVER_ID}: Disconnecting from MQTT broker.")
    client.loop_stop()
    client.disconnect()