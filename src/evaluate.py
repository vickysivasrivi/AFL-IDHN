# evaluate.py

import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Input, LSTM,
                                     Reshape)
from tensorflow.keras.models import Model, Sequential

# --- Configuration ---
DATA_FILE_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "Preprocessed",
    "cicids2017_preprocessed.csv",
)
N_FEATURES: int = 78
NUM_CLASSES: int = 15
BATCH_SIZE: int = 256

CLASS_NAMES = [
    'BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
    'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection',
    'Web Attack - XSS'
]


def build_model(n_features: int, num_classes: int) -> Model:
    """Builds the unified CNN-LSTM model."""
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


def load_test_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and returns the test dataset split."""
    print(f"Loading full dataset from: {file_path}")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    X = df.drop("label", axis=1)
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_reshaped = np.asarray(X_test, dtype=np.float32).reshape(
        (X_test.shape[0], 1, N_FEATURES)
    )
    print(f"Test data loaded. Shape: {X_test_reshaped.shape}")
    return X_test_reshaped, y_test.values.astype(np.int32)


# --- NEW: Function to perform evaluation and RETURN the results ---
def evaluate_model(model_weights_path: str, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
    """Evaluates a model and returns its classification report as a dictionary.
    
    Args:
        model_weights_path: Path to the saved .weights.h5 file.
        test_data: A tuple containing X_test and y_test.
        
    Returns:
        A dictionary containing the parsed classification report.
    """
    X_test, y_test = test_data
    model = build_model(N_FEATURES, NUM_CLASSES)
    
    print(f"\n--- Evaluating Model: {os.path.basename(model_weights_path)} ---")
    
    try:
        model.load_weights(model_weights_path)
    except Exception as e:
        print(f"ERROR: Could not load model weights. Error: {e}")
        return {} # Return empty dict on failure
        
    print("Generating predictions...")
    y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Generate the report as a dictionary for easy parsing
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    
    print("Evaluation complete.")
    return report_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_filename.weights.h5>")
        sys.exit(1)
        
    model_filename = sys.argv[1]
    model_weights_path = os.path.join("", model_filename)

    if not os.path.exists(model_weights_path):
        print(f"FATAL ERROR: The model weights file was not found at '{model_weights_path}'")
    else:
        # Load data once for the standalone run
        test_data = load_test_data(DATA_FILE_PATH)
        X_test, y_test = test_data

        # Evaluate the model and get the report dictionary
        report = evaluate_model(model_weights_path, test_data)
        
        # To get the text output for the console, we need to make predictions again
        if report:
            print("\n" + "="*60)
            print("           Classification Report")
            print("="*60)

            # <<< --- DEFINITIVE FIX: Separate the model creation, loading, and prediction --- >>>
            # 1. Create a temporary model instance
            temp_model = build_model(N_FEATURES, NUM_CLASSES)
            # 2. Load the weights into it (this returns None)
            temp_model.load_weights(model_weights_path)
            # 3. Now, call predict on the model object itself
            y_pred_proba = temp_model.predict(X_test, batch_size=BATCH_SIZE)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Print the text-based report
            print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))