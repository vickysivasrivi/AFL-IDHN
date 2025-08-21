# live_feature_detector.py

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from scapy.all import sniff, IP, TCP, UDP
from threading import Thread, Lock
from collections import defaultdict

# --- Configuration ---
MODEL_WEIGHTS_PATH = "final_global_model_adaptive.weights.h5"
# The network interface to sniff on. Use `ifconfig` on your Pi to find this.
NETWORK_INTERFACE = "eth0" # Or "wlan0" for Wi-Fi
# How long to consider a flow "active" before analyzing it (in seconds)
FLOW_TIMEOUT_SECONDS = 5
# How often to check for and analyze timed-out flows
ANALYSIS_INTERVAL_SECONDS = 2

# Model Parameters (must match your training)
N_FEATURES = 78
NUM_CLASSES = 15
CLASS_NAMES = [
    'BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
    'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection',
    'Web Attack - XSS'
]

# This is the exact order of columns from the CIC-IDS2017 dataset, minus the label
FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


# Global data structures for tracking flows
active_flows = defaultdict(dict)
flows_lock = Lock()

def build_model(n_features, num_classes):
    """Builds the identical model architecture used for training."""
    inputs = Input(shape=(1, n_features))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=100, activation='relu')(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_flow_key(packet):
    """Creates a unique key for a network flow."""
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = packet[IP].proto
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None
        # Sort to ensure (A,B) is the same as (B,A)
        return tuple(sorted(((src_ip, src_port), (dst_ip, dst_port)))) + (proto,)
    return None

def process_packet(packet):
    """This function is called by Scapy for each captured packet."""
    global active_flows
    flow_key = get_flow_key(packet)
    if not flow_key:
        return

    timestamp = time.time()
    packet_len = len(packet)
    
    with flows_lock:
        if flow_key not in active_flows:
            # Start a new flow
            active_flows[flow_key] = {
                'packets': [],
                'start_time': timestamp,
                'last_seen': timestamp
            }
        
        # Add packet info to the flow
        active_flows[flow_key]['packets'].append({
            'timestamp': timestamp,
            'len': packet_len,
            'is_forward': packet[IP].src == flow_key[0][0] # Check if it's a forward packet
        })
        active_flows[flow_key]['last_seen'] = timestamp

def analyze_flows(model):
    """Periodically checks for timed-out flows and analyzes them."""
    global active_flows
    print("--- Flow Analyzer Thread Started ---")
    while True:
        time.sleep(ANALYSIS_INTERVAL_SECONDS)
        
        timed_out_keys = []
        now = time.time()
        
        with flows_lock:
            for key, flow in active_flows.items():
                if now - flow['last_seen'] > FLOW_TIMEOUT_SECONDS:
                    timed_out_keys.append(key)
            
            for key in timed_out_keys:
                flow_data = active_flows.pop(key)
                # Create a separate thread for feature extraction and prediction
                # to avoid blocking the analyzer.
                Thread(target=extract_and_predict, args=(flow_data, model, key)).start()

def extract_and_predict(flow_data, model, flow_key):
    """Extracts features from a flow and makes a prediction."""
    print(f"\nAnalyzing flow: {flow_key[0][0]}:{flow_key[0][1]} <-> {flow_key[1][0]}:{flow_key[1][1]}")

    # --- Feature Extraction (Simplified Subset) ---
    features = {}
    packets = flow_data['packets']
    if len(packets) < 2:
        return # Need at least two packets to calculate IAT stats

    flow_duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) * 1_000_000 # microseconds
    
    fwd_packets = [p for p in packets if p['is_forward']]
    bwd_packets = [p for p in packets if not p['is_forward']]

    fwd_lengths = [p['len'] for p in fwd_packets]
    bwd_lengths = [p['len'] for p in bwd_packets]

    timestamps = [p['timestamp'] for p in packets]
    inter_arrival_times = np.diff(timestamps) * 1_000_000 # microseconds

    features['Flow Duration'] = flow_duration
    features['Total Fwd Packets'] = len(fwd_packets)
    features['Total Backward Packets'] = len(bwd_packets)
    features['Total Length of Fwd Packets'] = sum(fwd_lengths)
    features['Total Length of Bwd Packets'] = sum(bwd_lengths)
    features['Fwd Packet Length Max'] = max(fwd_lengths) if fwd_lengths else 0
    features['Fwd Packet Length Min'] = min(fwd_lengths) if fwd_lengths else 0
    features['Fwd Packet Length Mean'] = np.mean(fwd_lengths) if fwd_lengths else 0
    features['Fwd Packet Length Std'] = np.std(fwd_lengths) if fwd_lengths else 0
    features['Bwd Packet Length Max'] = max(bwd_lengths) if bwd_lengths else 0
    features['Bwd Packet Length Min'] = min(bwd_lengths) if bwd_lengths else 0
    features['Bwd Packet Length Mean'] = np.mean(bwd_lengths) if bwd_lengths else 0
    features['Bwd Packet Length Std'] = np.std(bwd_lengths) if bwd_lengths else 0
    
    features['Flow IAT Mean'] = np.mean(inter_arrival_times)
    features['Flow IAT Std'] = np.std(inter_arrival_times)
    features['Flow IAT Max'] = np.max(inter_arrival_times)
    features['Flow IAT Min'] = np.min(inter_arrival_times)
    
    features['Flow Packets/s'] = len(packets) / (flow_duration / 1_000_000) if flow_duration > 0 else 0
    features['Flow Bytes/s'] = (sum(fwd_lengths) + sum(bwd_lengths)) / (flow_duration / 1_000_000) if flow_duration > 0 else 0

    # Create a DataFrame with the correct column order, filling missing with 0
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS).fillna(0)
    
    # --- Prediction ---
    feature_vector = np.asarray(feature_df, dtype=np.float32).reshape((1, 1, N_FEATURES))
    
    prediction_proba = model.predict(feature_vector)
    predicted_class_index = np.argmax(prediction_proba)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction_proba) * 100
    
    if predicted_class_name != 'BENIGN':
        print(f"!!! ALERT !!! Potential Attack Detected!")
        print(f"    Flow: {flow_key[0][0]}:{flow_key[0][1]} -> {flow_key[1][0]}:{flow_key[1][1]}")
        print(f"    Prediction: {predicted_class_name} (Confidence: {confidence:.2f}%)")
        print(f"    Packets/sec: {features['Flow Packets/s']:.2f}")
    else:
        print(f"Flow classified as BENIGN (Confidence: {confidence:.2f}%)")


def main():
    print("--- Live Intrusion Detection System Starting ---")
    
    # 1. Load the trained model
    print(f"Loading model weights from {MODEL_WEIGHTS_PATH}...")
    try:
        model = build_model(N_FEATURES, NUM_CLASSES)
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        return

    # 2. Start the flow analysis thread in the background
    analyzer_thread = Thread(target=analyze_flows, args=(model,), daemon=True)
    analyzer_thread.start()

    # 3. Start sniffing packets on the main thread (this is a blocking call)
    print(f"Starting network sniffer on interface '{NETWORK_INTERFACE}'...")
    print("Press Ctrl+C to stop.")
    try:
        sniff(iface=NETWORK_INTERFACE, prn=process_packet, store=0)
    except KeyboardInterrupt:
        print("\n--- Shutting Down ---")
    except Exception as e:
        print(f"\nAn error occurred during sniffing: {e}")
        print("Ensure you are running with sudo and the interface name is correct.")

if __name__ == "__main__":
    main()