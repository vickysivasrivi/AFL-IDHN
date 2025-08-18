# server_federated_mqtt.py

import os
import time
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
import socket
import numpy as np
import json

# Ensure this utility file is accessible and named correctly
from utils_mqtt_fl import (
    serialize_weights, deserialize_weights, serialize_message, deserialize_message,
    weighted_average_adaptive
)

# --- Configuration ---
BROKER_HOST = os.environ.get("MQTT_HOST", "192.168.0.250")
BROKER_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", "vignesh_2181")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "Vignesh@2181")

# Topics
TOPIC_CLIENT_READY = "afl/client/ready"
TOPIC_CLIENT_UPDATE = "afl/client/update"
TOPIC_GLOBAL_START = "afl/global/start_round"
TOPIC_GLOBAL_WEIGHTS_HEADER = "afl/global/weights/header"
TOPIC_GLOBAL_WEIGHTS_CHUNK = "afl/global/weights/chunk"

# Federated Learning Parameters
TOTAL_ROUNDS = 10
MIN_CLIENTS_PER_ROUND = 1
ROUND_TIMEOUT = 300 # 5 minutes

# Model Parameters
N_FEATURES = 78
NUM_CLASSES = 15

def build_model(n_features, num_classes):
    """Creates the CNN-LSTM model, identical to the client's."""
    inputs = Input(shape=(1, n_features))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=100, activation='relu')(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
class AggregationServer:
    def __init__(self):
        self.updates = []
        self.ready_clients = set()
        self.current_round = 0 # Start at 0, will be incremented to 1 for the first round
        self.is_connected = False # Connection status flag

        self.global_model = build_model(N_FEATURES, NUM_CLASSES)
        print("[SERVER] Initial global model created.")
        server_client_id = f"AFL_Aggregation_Server_{socket.gethostname()}_{np.random.randint(1000)}"
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=server_client_id, clean_session=True)
        
        if MQTT_USERNAME:
            self.mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_disconnect = self.on_disconnect
        self.mqttc.on_message = self.on_message
        self.mqttc.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[SERVER] Connected with rc={rc}")
            self.is_connected = True
            client.subscribe(TOPIC_CLIENT_READY)
            client.subscribe(TOPIC_CLIENT_UPDATE)
            print(f"[SERVER] Subscribed to '{TOPIC_CLIENT_READY}' and '{TOPIC_CLIENT_UPDATE}'")
        else:
            print(f"[SERVER] Failed to connect, return code {rc}")
            self.is_connected = False

    # <<< FIX: Update the function signature to accept the new arguments >>>
    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        print(f"[SERVER] Disconnected with rc={rc}. The loop will pause.")
        self.is_connected = False

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        
        try:
            payload_str = msg.payload.decode("utf-8")
        except UnicodeDecodeError:
            payload_str = None

        if topic == TOPIC_CLIENT_READY and payload_str:
            client_id = deserialize_message(payload_str).get("client_id")
            if client_id:
                self.ready_clients.add(client_id)
                print(f"[SERVER] Client ready: {client_id} ({len(self.ready_clients)} total)")

        elif topic == TOPIC_CLIENT_UPDATE and payload_str:
            try:
                data = deserialize_message(payload_str)
                if data.get("round") == self.current_round:
                    weights = deserialize_weights(data["weights"])
                    self.updates.append({"weights": weights, "num_samples": data.get("num_samples", 0), "meta": data.get("meta", {})})
                    print(f"[SERVER] Received update from {data['client_id']} for round {self.current_round} ({len(self.updates)}/{MIN_CLIENTS_PER_ROUND})")
                else:
                    print(f"[SERVER] Received stale update from {data['client_id']} for round {data.get('round')}. Ignoring.")
            except Exception as e:
                print(f"[SERVER] Error processing update: {e}")

    def start_new_round(self):
        print(f"[SERVER] Broadcasting start signal and weights for round {self.current_round}")
        weights = self.global_model.get_weights()
        
        header = {"round": self.current_round, "num_layers": len(weights)}
        self.mqttc.publish(TOPIC_GLOBAL_WEIGHTS_HEADER, json.dumps(header), qos=1, retain=False)
        print(f"[SERVER] Published weights header: {header}")
        
        # <<< --- DEFINITIVE FIX: Add a small delay between each message --- >>>
        # This prevents the public broker's rate limit from disconnecting us.
        for i, layer_weights in enumerate(weights):
            payload = serialize_weights([layer_weights])
            self.mqttc.publish(f"{TOPIC_GLOBAL_WEIGHTS_CHUNK}/{i}", payload, qos=1, retain=False)
            time.sleep(0.05) # Add a 50 millisecond delay

        print(f"[SERVER] Published {len(weights)} weight layers.")

        start_payload = serialize_message({"round": self.current_round})
        self.mqttc.publish(TOPIC_GLOBAL_START, start_payload, qos=1, retain=False)
        print(f"[SERVER] Published round start signal.")


    def aggregate_and_broadcast(self):
        new_weights = weighted_average_adaptive(self.updates)
        if new_weights:
            self.global_model.set_weights(new_weights)
            print("[SERVER] Global model updated successfully.")

    def run_server(self):
        self.mqttc.loop_start()
        
        print("[SERVER] Waiting for stable MQTT connection...")
        while not self.is_connected:
            time.sleep(1)
        
        print(f"[SERVER] Waiting for at least {MIN_CLIENTS_PER_ROUND} client(s) to connect...")
        while len(self.ready_clients) < MIN_CLIENTS_PER_ROUND:
            time.sleep(1)
        
        if not self.is_connected:
            print("[SERVER] Shutting down, lost connection before any clients were ready.")
            self.mqttc.loop_stop()
            return 
        
        print(f"[SERVER] {len(self.ready_clients)} client(s) ready. Starting rounds.")

        while self.current_round < TOTAL_ROUNDS:
            self.current_round += 1
            if not self.is_connected:
                print("[SERVER] Pausing due to MQTT disconnection...")
                time.sleep(5)
                continue

            print(f"\n[SERVER] = Round {self.current_round} start =")
            self.updates.clear()
            self.start_new_round()

            start_time = time.time()
            while len(self.updates) < MIN_CLIENTS_PER_ROUND and (time.time() - start_time) < ROUND_TIMEOUT:
                time.sleep(1)

            if len(self.updates) >= MIN_CLIENTS_PER_ROUND:
                print(f"[SERVER] Aggregating {len(self.updates)} updates for round {self.current_round}")
                self.aggregate_and_broadcast()
            else:
                print(f"[SERVER] Round {self.current_round} timed out. Insufficient updates ({len(self.updates)}).")
            
            time.sleep(5)

        print("[SERVER] All rounds completed. Shutting down.")
        self.mqttc.loop_stop()

if __name__ == "__main__":
    server = AggregationServer()
    try:
        # The method is now correctly part of the server object
        server.run_server()
    except KeyboardInterrupt:
        print("\n[SERVER] Manual shutdown.")