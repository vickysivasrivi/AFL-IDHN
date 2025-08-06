import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import paho.mqtt.client as paho
from paho import mqtt
import time
import ssl
import numpy as np
import json # For simple serialization
import pickle # For PyTorch model state_dict serialization

# --- Configuration for MQTT and Federated Learning ---
CLIENT_ID = "fl_client_1" # <<< IMPORTANT: Change this for each Raspberry Pi!
MQTT_BROKER_HOST = "2e09a248750440018083de7318e45187.s1.eu.hivemq.cloud" # Your HiveMQ Cloud Hostname
MQTT_BROKER_PORT = 8883 # Standard TLS port for HiveMQ Cloud
MQTT_USERNAME = "vignesh_2181" # <<< IMPORTANT: Replace with your HiveMQ username
MQTT_PASSWORD = "Vignesh@2181" # <<< IMPORTANT: Replace with your HiveMQ password

MQTT_TLS_CA_CERT = "hivemq-ca.pem" # Path to your downloaded HiveMQ CA certificate

# MQTT Topics
GLOBAL_MODEL_TOPIC = "fl/global_model" # Server publishes to this
LOCAL_UPDATE_TOPIC_PREFIX = "fl/updates/" # Clients publish to fl/updates/<CLIENT_ID>

# Model configuration (should match fl_server.py)
INPUT_FEATURES = 78 # Number of features in your preprocessed data
NUM_CLASSES = 15   # Number of classes for your labels

# --- Helper Functions for Model Serialization/Deserialization ---
def serialize_model_weights(model_state_dict):
    """Serializes PyTorch model state_dict for transmission."""
    return pickle.dumps(model_state_dict)

def deserialize_model_weights(serialized_weights):
    """Deserializes PyTorch model state_dict from received bytes."""
    return pickle.loads(serialized_weights)


# --- 1. PyTorch CNN-LSTM Model ---
# This function defines the model using PyTorch's `torch.nn` modules.
class CNNDetector(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CNNDetector, self).__init__()
        # PyTorch Conv1d expects input of shape (batch, channels, length)
        # We will treat each feature as a separate channel
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
        
        # Calculate the size of the tensor after CNN layers to pass to LSTM
        # Input features (L) -> L / 2 -> L / 2 / 2
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
        # Input x is (batch_size, input_features). We need to reshape it for Conv1d.
        x = x.unsqueeze(1) # Reshape to (batch_size, 1, input_features)
        x = self.cnn(x)
        
        # Reshape output of CNN for LSTM
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = x.unsqueeze(1) # Add a 'timestep' dimension for LSTM: (batch_size, 1, flattened_features)
        
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x.squeeze(1))
        
        x = self.fc(x)
        return x

# Initialize the PyTorch model for the client
local_model = CNNDetector(input_features=INPUT_FEATURES, num_classes=NUM_CLASSES)
print(f"Client {CLIENT_ID}: Local CNN-LSTM model initialized.")


# --- Dummy Local Data (for initial testing) ---
# <<< IMPORTANT: This needs to be replaced by actual partitioned, preprocessed data from Developer A.
def load_local_data(client_id, num_samples=100):
    """Loads dummy local data for a client as a PyTorch DataLoader."""
    print(f"Client {client_id}: Loading dummy local data ({num_samples} samples)...")
    features = torch.tensor(np.random.rand(num_samples, INPUT_FEATURES).astype(np.float32))
    labels = torch.tensor(np.random.randint(0, NUM_CLASSES, num_samples).astype(np.int64))
    
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

client_local_data = load_local_data(CLIENT_ID, num_samples=200)


# --- MQTT Callbacks ---

def on_connect(client, userdata, flags, rc, properties=None):
    print(rc)
    if rc == 0:
        print(f"Client {CLIENT_ID}: Connected to MQTT Broker (HiveMQ) successfully!")
        client.subscribe(GLOBAL_MODEL_TOPIC, qos=1)
        print(f"Client {CLIENT_ID}: Subscribed to '{GLOBAL_MODEL_TOPIC}'")
    else:
        print(f"Client {CLIENT_ID}: Failed to connect, return code {rc}\n")

def on_publish(client, userdata, mid, properties=None):
    print(f"Client {CLIENT_ID}: Published message ID: {str(mid)}")

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print(f"Client {CLIENT_ID}: Subscribed: {str(mid)} QoS: {str(granted_qos)}")

def on_message(client, userdata, msg):
    print(f"Client {CLIENT_ID}: Message received on topic '{msg.topic}'")
    if msg.topic == GLOBAL_MODEL_TOPIC:
        print(f"Client {CLIENT_ID}: Received global model. Starting local training...")
        
        try:
            # 1. Deserialize the global model weights (state_dict)
            global_state_dict = deserialize_model_weights(msg.payload)
            local_model.load_state_dict(global_state_dict)
            print(f"Client {CLIENT_ID}: Global model loaded into local model.")

            # 2. Perform local training with PyTorch loop
            # Define loss and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(local_model.parameters(), lr=0.01)
            
            local_model.train() # Set model to training mode
            total_loss = 0
            for features, labels in client_local_data:
                optimizer.zero_grad()
                outputs = local_model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(client_local_data)
            print(f"Client {CLIENT_ID}: Local training complete. Average Loss: {avg_loss:.4f}")

            # 3. Get local model updates (the updated state_dict)
            local_updated_state_dict = local_model.state_dict()
            
            # Send the number of examples this client trained on (for client-aware weighting)
            num_local_examples = len(client_local_data.dataset)

            # 4. Serialize local updates and metadata
            client_update_payload = {
                'state_dict': local_updated_state_dict,
                'num_examples': num_local_examples
            }
            serialized_local_update = pickle.dumps(client_update_payload)
            
            # 5. Publish local updates
            publish_topic = f"{LOCAL_UPDATE_TOPIC_PREFIX}{CLIENT_ID}"
            client.publish(publish_topic, payload=serialized_local_update, qos=1)
            print(f"Client {CLIENT_ID}: Published local model update to '{publish_topic}' (Num Examples: {num_local_examples})")

        except Exception as e:
            print(f"Client {CLIENT_ID}: Error during model processing or local training: {e}")

# --- MQTT Client Setup ---
client = paho.Client(client_id=CLIENT_ID, userdata=None, protocol=paho.MQTTv5)
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.on_publish = on_publish

try:
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    print(f"Client {CLIENT_ID}: TLS enabled with CA certificate: {MQTT_TLS_CA_CERT}")
except FileNotFoundError:
    print(f"Client {CLIENT_ID}: Error: CA certificate not found at {MQTT_TLS_CA_CERT}.")
    print("Please download it from HiveMQ Cloud and ensure the path is correct.")
    exit()

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

print(f"Client {CLIENT_ID}: Attempting to connect to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Client {CLIENT_ID}: MQTT connection failed: {e}")
    print("Please check your network, broker hostname, port, username, and password.")
    exit()

client.loop_start()

print(f"Client {CLIENT_ID}: Waiting for global model from '{GLOBAL_MODEL_TOPIC}'...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print(f"Client {CLIENT_ID}: Disconnecting from MQTT broker.")
    client.loop_stop()
    client.disconnect()