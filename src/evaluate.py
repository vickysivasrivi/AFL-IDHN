# evaluate.py

import os
import sys # Import the sys module to read command-line arguments
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
# Path to your FULL preprocessed dataset
DATA_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'Preprocessed', 'cicids2017_preprocessed.csv'
)
# Model Parameters (Must be identical to your training scripts)
N_FEATURES = 78
NUM_CLASSES = 15

def build_model(n_features, num_classes):
    """Creates the CNN-LSTM model, identical to the training models."""
    inputs = Input(shape=(1, n_features))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=100, activation='relu')(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_test_data(file_path):
    """
    Loads the full dataset and splits it, returning only the test set.
    """
    print(f"Loading full dataset from: {file_path}")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_reshaped = np.asarray(X_test).reshape((X_test.shape[0], 1, N_FEATURES))
    
    print(f"Test data loaded. Shape: {X_test_reshaped.shape}")
    return X_test_reshaped, y_test.values.astype(np.int32)

def evaluate_model(model_weights_path):
    print("\n--- Starting Evaluation ---")
    
    X_test, y_test = load_test_data(DATA_FILE_PATH)
    model = build_model(N_FEATURES, NUM_CLASSES)
    
    try:
        print(f"Loading saved model weights from: {model_weights_path}")
        model.load_weights(model_weights_path)
    except Exception as e:
        print(f"ERROR: Could not load model weights. Error: {e}")
        return
        
    print("Evaluating model performance on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # Set verbose=0 to keep the report clean
    
    print(f"\n--- Evaluation Results for {model_weights_path} ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nGenerating detailed classification report...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_test, y_pred, digits=4))
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("FATAL ERROR: You must provide the path to the model weights file.")
        print("Usage: python evaluate.py <path_to_model.weights.h5>")
        sys.exit(1) # Exit the script if no path is given
        
    model_weights_path = sys.argv[1]

    if not os.path.exists(model_weights_path):
        print(f"FATAL ERROR: The model weights file was not found at '{model_weights_path}'")
    else:
        evaluate_model(model_weights_path)