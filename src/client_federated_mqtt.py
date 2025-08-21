# client_federated_mqtt.py

import os
import json
import time
import socket
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Reshape
from typing import Dict, Any, Tuple

# Ensure this utility file is accessible
from utils_mqtt_fl import (
    serialize_weights,
    deserialize_weights,
    serialize_message,
    deserialize_message,
)

# --- Configuration ---
BROKER_HOST: str = os.environ.get("MQTT_HOST", "192.168.0.250")
BROKER_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME: str = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD: str = os.environ.get("MQTT_PASSWORD", "")
CLIENT_ID: str = f"client_{socket.gethostname()}_{np.random.randint(1000)}"
DATA_FILE_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "Preprocessed",
    "cicids2017_preprocessed.csv",
)
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "256"))
LOCAL_EPOCHS: int = int(os.environ.get("LOCAL_EPOCHS", "1"))

# --- MQTT Topics ---
TOPIC_CLIENT_READY: str = "afl/client/ready"
TOPIC_CLIENT_UPDATE: str = "afl/client/update"
TOPIC_GLOBAL_START: str = "afl/global/start_round"
TOPIC_GLOBAL_WEIGHTS_HEADER: str = "afl/global/weights/header"
TOPIC_GLOBAL_WEIGHTS_CHUNK: str = "afl/global/weights/chunk"

# --- Model Parameters (must match server) ---
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


def load_local_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Loads and prepares the local dataset for training."""
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found at: {os.path.abspath(full_path)}")

    df = pd.read_csv(full_path)
    df.dropna(inplace=True)
    X = df.drop("label", axis=1)
    y = df["label"]
    n_features = X.shape[1]
    # Reshape features to (samples, 1, features) for Conv1D and LSTM
    X_reshaped = np.asarray(X, dtype=np.float32).reshape((X.shape[0], 1, n_features))
    return (X_reshaped, y.values.astype(np.int32), n_features)


class FLClient:
    """Handles the client-side logic for federated learning."""

    def __init__(self) -> None:
        """Initializes the client."""
        (self.X, self.y, self.n_features) = load_local_data(DATA_FILE_PATH)
        self.local_model: Model = build_model(self.n_features, NUM_CLASSES)
        self.current_round: int = 0
        self.weight_layers: Dict[int, np.ndarray] = {}
        self.expected_layers: int = 0

        self.mqttc = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID, clean_session=True
        )
        if MQTT_USERNAME:
            self.mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        self.mqttc.on_connect = self._on_connect
        self.mqttc.on_message = self._on_message
        self.mqttc.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """Callback for successful MQTT connection."""
        print(f"[{CLIENT_ID}] Connected with rc={rc}")
        client.subscribe(TOPIC_GLOBAL_START)
        client.subscribe(TOPIC_GLOBAL_WEIGHTS_HEADER)
        client.subscribe(f"{TOPIC_GLOBAL_WEIGHTS_CHUNK}/#")  # Subscribe to all chunk topics
        ready = {"client_id": CLIENT_ID, "status": "ready"}
        client.publish(TOPIC_CLIENT_READY, serialize_message(ready), qos=1)

    def _on_message(self, client, userdata, msg) -> None:
        """Callback to handle incoming messages from the server."""
        topic = msg.topic
        if topic == TOPIC_GLOBAL_WEIGHTS_HEADER:
            try:
                header = json.loads(msg.payload.decode("utf-8"))
                self.expected_layers = header.get("num_layers", 0)
                self.weight_layers.clear()
            except Exception as e:
                print(f"[{CLIENT_ID}] Error decoding header: {e}")

        elif topic.startswith(TOPIC_GLOBAL_WEIGHTS_CHUNK):
            try:
                layer_index = int(topic.split("/")[-1])
                deserialized_layer = deserialize_weights(msg.payload.decode("utf-8"))
                self.weight_layers[layer_index] = deserialized_layer[0]
            except Exception as e:
                print(f"[{CLIENT_ID}] Error processing layer: {e}")

        elif topic == TOPIC_GLOBAL_START:
            round_num = int(deserialize_message(msg.payload).get("round", 0))
            if round_num > self.current_round:
                self.current_round = round_num
                print(f"[{CLIENT_ID}] Received start signal for round {self.current_round}")
                self._check_and_train()

    def _check_and_train(self) -> None:
        """Checks if all weight layers are received and then starts training."""
        if self.expected_layers > 0 and len(self.weight_layers) == self.expected_layers:
            print(f"[{CLIENT_ID}] All {self.expected_layers} layers received. Applying weights.")
            try:
                reassembled_weights = [
                    self.weight_layers[i] for i in sorted(self.weight_layers.keys())
                ]
                self.local_model.set_weights(reassembled_weights)
                print(f"[{CLIENT_ID}] Starting local training.")
                self._train_and_update()
            except Exception as e:
                print(f"[{CLIENT_ID}] Error during weight processing: {e}")
        else:
            print(f"[{CLIENT_ID}] Waiting for more weight layers...")

    def _train_and_update(self) -> None:
        """Performs local training and publishes the updated model and metrics."""
        history = self.local_model.fit(
            self.X, self.y, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=1
        )
        final_loss = history.history["loss"][-1]
        updated_weights = self.local_model.get_weights()
        payload = {
            "client_id": CLIENT_ID,
            "round": self.current_round,
            "num_samples": len(self.X),
            "loss": final_loss,
            "meta": {},
            "weights": serialize_weights(updated_weights),
        }
        self.mqttc.publish(TOPIC_CLIENT_UPDATE, serialize_message(payload), qos=1)
        print(
            f"[{CLIENT_ID}] Published update for round {self.current_round} with loss: {final_loss:.4f}"
        )

    def loop_forever(self) -> None:
        """Starts the main client loop."""
        self.mqttc.loop_forever()


if __name__ == "__main__":
    # Suppress TensorFlow informational messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    try:
        client = FLClient()
        client.loop_forever()
    except Exception as e:
        print(f"\n[CLIENT] FATAL ERROR: {e}\n")