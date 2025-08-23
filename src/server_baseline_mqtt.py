#!/usr/bin/env python3

"""
Federated Learning Aggregation Server (Baseline)

Author: Vignesh Siva
Date: August 2025
Version: 2.0

This script implements a baseline federated learning server. It uses a simple 
weighted averaging strategy based on the number of samples each client used for 
training. It communicates with clients via MQTT, aggregates updates, and saves 
the final global model after all rounds are complete.
"""


import os
import time
import json
import socket
import threading
from typing import Dict, Any, List

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Reshape

# --- IMPORTANT: Import the baseline (non-adaptive) aggregation function ---
from utils_mqtt_fl import (
    serialize_weights,
    deserialize_weights,
    serialize_message,
    deserialize_message,
    weighted_average_adaptive,
)

# --- Configuration ---
BROKER_HOST: str = os.environ.get("MQTT_HOST", "192.168.0.250")
BROKER_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME: str = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD: str = os.environ.get("MQTT_PASSWORD", "")
SERVER_ID: str = f"AFL_Baseline_Server_{socket.gethostname()}"

# --- MQTT Topics ---
TOPIC_CLIENT_READY: str = "afl/client/ready"
TOPIC_CLIENT_UPDATE: str = "afl/client/update"
TOPIC_GLOBAL_START: str = "afl/global/start_round"
TOPIC_GLOBAL_WEIGHTS_HEADER: str = "afl/global/weights/header"
TOPIC_GLOBAL_WEIGHTS_CHUNK: str = "afl/global/weights/chunk"

# --- Federated Learning Parameters ---
TOTAL_ROUNDS: int = 10
MIN_CLIENTS_PER_ROUND: int = 1
ROUND_TIMEOUT: int = 120  # 2 minutes
N_FEATURES: int = 78
NUM_CLASSES: int = 15


# --- UNIFIED TensorFlow/Keras Model Definition ---
def build_model(n_features: int, num_classes: int) -> Model:
    """Creates a lightweight CNN-LSTM model."""
    model_input = Input(shape=(1, n_features))
    x = Conv1D(filters=64, kernel_size=1, activation="relu")(model_input)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=100, activation="relu")(x)
    model_output = Dense(units=num_classes, activation="softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


class AggregationServer:
    """Orchestrates the baseline federated learning process."""

    def __init__(self) -> None:
        """Initializes the server, model, and MQTT client."""
        self.updates: List[Dict[str, Any]] = []
        self.ready_clients: set = set()
        self.current_round: int = 0
        self.is_connected: bool = False
        self.global_model: Model = build_model(N_FEATURES, NUM_CLASSES)
        print("[SERVER - BASELINE] Initial global model created.")

        self.mqttc = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=SERVER_ID, clean_session=True
        )
        if MQTT_USERNAME:
            self.mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        self.mqttc.on_connect = self._on_connect
        self.mqttc.on_disconnect = self._on_disconnect
        self.mqttc.on_message = self._on_message
        self.mqttc.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for successful MQTT connection."""
        if rc == 0:
            print(f"[SERVER - BASELINE] Connected to MQTT Broker")
            self.is_connected = True
            client.subscribe(TOPIC_CLIENT_READY); client.subscribe(TOPIC_CLIENT_UPDATE)
            print(f"[SERVER - BASELINE] Subscribed to client topics.")
        else:
            print(f"[SERVER - BASELINE] Failed to connect, return code {rc}")
            self.is_connected = False

    def _on_disconnect(self, client, userdata, flags,rc, properties=None):
        """Callback for MQTT disconnection."""
        print(f"[SERVER - BASELINE] Disconnected from broker.")
        self.is_connected = False

    def _on_message(self, client, userdata, msg):
        """Callback to handle incoming messages from clients."""
        topic = msg.topic
        try:
            payload_str = msg.payload.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return

        if topic == TOPIC_CLIENT_READY:
            client_id = deserialize_message(payload_str).get("client_id")
            if client_id: self.ready_clients.add(client_id); print(f"[SERVER - BASELINE] Client ready: {client_id} ({len(self.ready_clients)} total)")
        elif topic == TOPIC_CLIENT_UPDATE:
            try:
                data = deserialize_message(payload_str)
                if data.get("round") == self.current_round:
                    weights = deserialize_weights(data["weights"])
                    self.updates.append(
                        {
                            "weights": weights,
                            "num_samples": data.get("num_samples", 0),
                            "loss": data.get("loss", 1.0),
                        }
                    )
                    print(f"[SERVER - BASELINE] Received update from {data['client_id']}")
                else:
                    print(f"[SERVER - BASELINE] Received stale update. Ignoring.")
            except Exception as e:
                print(f"[SERVER - BASELINE] Error processing update: {e}")

    def start_new_round(self) -> None:
        """Broadcasts the start signal and model weights."""
        weights = self.global_model.get_weights()
        header = {"round": self.current_round, "num_layers": len(weights)}
        self.mqttc.publish(TOPIC_GLOBAL_WEIGHTS_HEADER, json.dumps(header), qos=1)

        for i, layer_weights in enumerate(weights):
            payload = serialize_weights([layer_weights])
            self.mqttc.publish(f"{TOPIC_GLOBAL_WEIGHTS_CHUNK}/{i}", payload, qos=1)
            time.sleep(0.05)

        start_payload = serialize_message({"round": self.current_round})
        self.mqttc.publish(TOPIC_GLOBAL_START, start_payload, qos=1)
        print(f"[SERVER - BASELINE] Published model and start signal for round {self.current_round}.")

    def aggregate_updates(self) -> None:
        """Aggregates client updates using the baseline (sample-size) method."""
        if not self.updates:
            print("[SERVER - BASELINE] No updates to aggregate.")
            return

        # --- KEY DIFFERENCE: Use the simple weighted average ---
        new_weights = weighted_average_adaptive(self.updates)

        if new_weights:
            self.global_model.set_weights(new_weights)
            print("[SERVER - BASELINE] Global model updated.")

    def run_server(self) -> None:
        """Main server loop."""
        self.mqttc.loop_start()
        print("[SERVER - BASELINE] Waiting for MQTT connection...")
        while not self.is_connected: time.sleep(1)
        print(f"[SERVER - BASELINE] Waiting for {MIN_CLIENTS_PER_ROUND} client(s)...")
        while len(self.ready_clients) < MIN_CLIENTS_PER_ROUND: time.sleep(1)

        for r in range(1, TOTAL_ROUNDS + 1):
            self.current_round = r
            print(f"\n===== Round {self.current_round}/{TOTAL_ROUNDS} =====")
            self.updates.clear()
            self.start_new_round()

            print(f"[SERVER - BASELINE] Waiting for updates ({ROUND_TIMEOUT}s)...")
            time.sleep(ROUND_TIMEOUT)

            if len(self.updates) >= MIN_CLIENTS_PER_ROUND:
                self.aggregate_updates()
            else:
                print(f"[SERVER - BASELINE] Round timed out.")

            time.sleep(5)

        # Save the final model weights after all rounds
        print("\n[SERVER - BASELINE] All rounds completed. Saving final model...")
        save_path = os.path.join("saved_models", "final_global_model_baseline.weights.h5")
        self.global_model.save_weights(save_path)
        print(f"[SERVER - BASELINE] Model saved to '{save_path}'. Shutting down.")

        self.mqttc.loop_stop()
        self.mqttc.disconnect()


if __name__ == "__main__":
    server = AggregationServer()
    try:
        server.run_server()
    except KeyboardInterrupt:
        print("\n[SERVER - BASELINE] Manual shutdown.")