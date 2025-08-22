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
import smtplib
from email.message import EmailMessage
import paho.mqtt.client as mqtt
import os
import socket
import json
from datetime import datetime

# --- Configuration ---
MODEL_WEIGHTS_PATH = "saved_models/final_global_model_adaptive.weights.h5"
NETWORK_INTERFACE = "wlan0" # Or "eth0" for Wi-Fi
FLOW_TIMEOUT_SECONDS = 5
ANALYSIS_INTERVAL_SECONDS = 2

alert_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"alert_client_{socket.gethostname()}")
# Connect to your broker
alert_client.connect("192.168.0.250", 1883, 60)
alert_client.loop_start() 

# <<< NEW: Configuration for Sustained Alerting >>>
# The duration an attack must last to trigger the significant alert
SUSTAINED_ATTACK_THRESHOLD_SECONDS = 10 
# How long to wait before "forgetting" an attacker after their last known activity
ATTACKER_COOLDOWN_SECONDS = 60

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


# Global data structures
active_flows = defaultdict(dict)
flows_lock = Lock()

# <<< NEW: Global tracker for sustained attacks >>>
sustained_attack_tracker = {}
tracker_lock = Lock()

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
        return tuple(sorted(((src_ip, src_port), (dst_ip, dst_port)))) + (proto,)
    return None

def process_packet(packet):
    """This function is called by Scapy for each captured packet."""
    global active_flows
    flow_key = get_flow_key(packet)
    if not flow_key: return

    timestamp = time.time()
    packet_len = len(packet)
    
    with flows_lock:
        if flow_key not in active_flows:
            active_flows[flow_key] = {'packets': [], 'start_time': timestamp, 'last_seen': timestamp}
        
        flow_info = {'timestamp': timestamp, 'len': packet_len}
        # <<< NEW: Determine which IP is the source for this specific packet >>>
        if IP in packet:
            flow_info['src_ip'] = packet[IP].src
        active_flows[flow_key]['packets'].append(flow_info)
        active_flows[flow_key]['last_seen'] = timestamp

def analyze_flows(model):
    """Periodically checks for timed-out flows and analyzes them."""
    global active_flows, sustained_attack_tracker
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
                Thread(target=extract_and_predict, args=(flow_data, model, key)).start()

        # <<< NEW: Periodic cleanup of the sustained attack tracker >>>
        with tracker_lock:
            stale_attackers = []
            for ip, data in sustained_attack_tracker.items():
                if now - data['last_seen'] > ATTACKER_COOLDOWN_SECONDS:
                    stale_attackers.append(ip)
            
            for ip in stale_attackers:
                del sustained_attack_tracker[ip]
                print(f"[*] Attacker {ip} has gone quiet. Removing from tracker.")

def extract_and_predict(flow_data, model, flow_key):
    """Extracts features from a flow and makes a prediction."""
    packets = flow_data['packets']
    if len(packets) < 2: return

    # --- Feature Extraction (Simplified Subset) ---
    features = {}
    flow_duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) * 1_000_000
    fwd_packets = [p for p in packets if p.get('src_ip') == flow_key[0][0]]
    bwd_packets = [p for p in packets if p.get('src_ip') != flow_key[0][0]]
    fwd_lengths = [p['len'] for p in fwd_packets]; bwd_lengths = [p['len'] for p in bwd_packets]
    timestamps = [p['timestamp'] for p in packets]; inter_arrival_times = np.diff(timestamps) * 1_000_000
    features['Flow Duration'] = flow_duration
    features['Total Fwd Packets'] = len(fwd_packets); features['Total Backward Packets'] = len(bwd_packets)
    features['Total Length of Fwd Packets'] = sum(fwd_lengths); features['Total Length of Bwd Packets'] = sum(bwd_lengths)
    features['Fwd Packet Length Max'] = max(fwd_lengths) if fwd_lengths else 0; features['Fwd Packet Length Min'] = min(fwd_lengths) if fwd_lengths else 0
    features['Fwd Packet Length Mean'] = np.mean(fwd_lengths) if fwd_lengths else 0; features['Fwd Packet Length Std'] = np.std(fwd_lengths) if fwd_lengths else 0
    features['Bwd Packet Length Max'] = max(bwd_lengths) if bwd_lengths else 0; features['Bwd Packet Length Min'] = min(bwd_lengths) if bwd_lengths else 0
    features['Bwd Packet Length Mean'] = np.mean(bwd_lengths) if bwd_lengths else 0; features['Bwd Packet Length Std'] = np.std(bwd_lengths) if bwd_lengths else 0
    features['Flow IAT Mean'] = np.mean(inter_arrival_times); features['Flow IAT Std'] = np.std(inter_arrival_times)
    features['Flow IAT Max'] = np.max(inter_arrival_times); features['Flow IAT Min'] = np.min(inter_arrival_times)
    features['Flow Packets/s'] = len(packets) / (flow_duration / 1_000_000) if flow_duration > 0 else 0
    features['Flow Bytes/s'] = (sum(fwd_lengths) + sum(bwd_lengths)) / (flow_duration / 1_000_000) if flow_duration > 0 else 0
    
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS).fillna(0)
    
    # --- Prediction ---
    feature_vector = np.asarray(feature_df, dtype=np.float32).reshape((1, 1, N_FEATURES))
    prediction_proba = model.predict(feature_vector, verbose=0)
    predicted_class_index = np.argmax(prediction_proba)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction_proba) * 100
    
    # <<< NEW: Sustained Alerting Logic >>>
    if predicted_class_name != 'BENIGN':
        # Identify the attacker IP (the one sending forward packets in this flow)
        attacker_ip = flow_key[0][0] if fwd_packets else flow_key[1][0]
        now = time.time()
        
        with tracker_lock:
            if attacker_ip not in sustained_attack_tracker:
                # First time we see this attacker
                sustained_attack_tracker[attacker_ip] = {'first_seen': now, 'last_seen': now, 'alert_triggered': False}
                print(f"\n[*] Initial suspicious activity ({predicted_class_name}) detected from {attacker_ip}.\n")
            else:
                # We are already tracking this attacker
                tracker = sustained_attack_tracker[attacker_ip]
                tracker['last_seen'] = now
                duration = now - tracker['first_seen']
                
                # Check if the attack has been sustained long enough and we haven't alerted yet
                if duration > SUSTAINED_ATTACK_THRESHOLD_SECONDS and not tracker['alert_triggered']:
                    print("\n" + "="*60)
                    print(f"!!! SUSTAINED ATTACK ALERT !!!")
                    print(f"    Attacker IP:  {attacker_ip}")
                    print(f"    Attack Type:  {predicted_class_name}")
                    print(f"    Confidence:   {confidence:.2f}%")
                    print(f"    Sustained for: {duration:.2f} seconds")
                    print("="*60 + "\n")
                    # Set the flag to prevent this from printing again for this instance
                    tracker['alert_triggered'] = True
                    send_email_alert(attacker_ip, predicted_class_name, confidence, duration)
                    alert_topic = "afl/alerts/sustained_attack"
                    alert_payload = {
                        "client_hostname": socket.gethostname(),
                        "interface": NETWORK_INTERFACE,
                        "attacker_ip": attacker_ip,
                        "attack_type": predicted_class_name,
                        "confidence": float(round(confidence, 2)),
                        "duration_sec": float(round(duration, 2)),
                        "timestamp": time.time()
                    }
                    
                    print(f"[*] Publishing alert to MQTT topic: {alert_topic}")
                    alert_client.publish(alert_topic, json.dumps(alert_payload))

    else:
        print(f"Flow classified as BENIGN (Confidence: {confidence:.2f}%)")


def send_email_alert(attacker_ip, attack_type, confidence, duration):
    """Sends a formatted email alert."""
    
    # --- Configuration ---
    # The recipient's email address
    email_recipient = "vickysivasrivi@gmail.com"
    
    # Get credentials from environment variables for security
    email_user = os.environ.get('EMAIL_USER')
    email_password = os.environ.get('EMAIL_PASSWORD')
    
    if not all([email_user, email_password]):
        print("[!] Email credentials not set. Cannot send email alert.")
        return

    # --- Create the Email ---
    msg = EmailMessage()
    msg['Subject'] = f"Sustained Attack Detected on {socket.gethostname()}"
    msg['From'] = email_user
    msg['To'] = email_recipient
    
    if confidence > 95:
        confidence_bg_color = "#D9534F" # Red for high confidence
    elif confidence > 75:
        confidence_bg_color = "#F0AD4E" # Orange for medium confidence
    else:
        confidence_bg_color = "#777777" # Gray for lower confidence

    # Create the email body
    body = f"""
    A sustained attack has been detected on your Raspberry Pi.
    
    --- Client Details ---
    Hostname: {socket.gethostname()}
    Network Interface: {NETWORK_INTERFACE}
    
    --- Attack Details ---
    Attacker IP: {attacker_ip}
    Attack Type: {attack_type}
    Confidence: {confidence:.2f}%
    Sustained for: {duration:.2f} seconds
    
    Please investigate this activity.
    """
    html_body = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Sustained Attack Alert</title></head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden;">
            <div style="background-color: #4A5568; color: #ffffff; padding: 20px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px;">Intrusion Detection System Alert</h1>
            </div>
            <div style="padding: 20px 30px;">
                <h2 style="color: #D9534F; font-size: 28px; text-align: center; margin-top: 0;">&#9888; Sustained Attack Detected</h2>
                <p style="color: #555555; font-size: 16px; line-height: 1.6;">A persistent threat has been identified on one of your network clients. Please review the details below and take immediate action.</p>
                <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 20px 0;">
                <h3 style="color: #333333; border-bottom: 2px solid #4A5568; padding-bottom: 5px;">Client Details</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <tr style="border-bottom: 1px solid #eeeeee;"><td style="padding: 10px 0; color: #555555; font-weight: bold; width: 150px;">Hostname:</td><td style="padding: 10px 0; color: #333333;">{socket.gethostname()}</td></tr>
                    <tr style="border-bottom: 1px solid #eeeeee;"><td style="padding: 10px 0; color: #555555; font-weight: bold;">Interface:</td><td style="padding: 10px 0; color: #333333;">{NETWORK_INTERFACE}</td></tr>
                </table>
                <h3 style="color: #333333; border-bottom: 2px solid #D9534F; padding-bottom: 5px; margin-top: 30px;">Attack Details</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <tr style="border-bottom: 1px solid #eeeeee;"><td style="padding: 10px 0; color: #555555; font-weight: bold; width: 150px;">Attacker IP:</td><td style="padding: 10px 0; color: #D9534F; font-weight: bold; font-size: 18px;">{attacker_ip}</td></tr>
                    <tr style="border-bottom: 1px solid #eeeeee;"><td style="padding: 10px 0; color: #555555; font-weight: bold;">Attack Type:</td><td style="padding: 10px 0; color: #333333;">{attack_type}</td></tr>
                    <tr style="border-bottom: 1px solid #eeeeee;"><td style="padding: 10px 0; color: #555555; font-weight: bold;">Confidence:</td><td style="padding: 10px 0; color: #333333;"><span style="background-color: {confidence_bg_color}; color: #ffffff; padding: 5px 10px; border-radius: 5px; font-size: 14px; font-weight: bold;">{confidence:.2f}%</span></td></tr>
                    <tr><td style="padding: 10px 0; color: #555555; font-weight: bold;">Sustained For:</td><td style="padding: 10px 0; color: #333333;">{duration:.2f} seconds</td></tr>
                </table>
            </div>
            <div style="background-color: #f4f4f4; color: #888888; text-align: center; padding: 15px; font-size: 12px;">
                <p style="margin: 0;">This is an automated alert. Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
            </div>
        </div>
    </body>
    </html>"""
    msg.set_content(body)
    msg.add_alternative(html_body, subtype='html')

    # --- Send the Email ---
    try:
        print("[*] Sending email alert...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_user, email_password)
            smtp.send_message(msg)
        print("[*] Email alert sent successfully.")
    except Exception as e:
        print(f"[!] FAILED to send email alert: {e}")


def main():
    print("--- Live Intrusion Detection System Starting ---")
    try:
        model = build_model(N_FEATURES, NUM_CLASSES)
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        return

    analyzer_thread = Thread(target=analyze_flows, args=(model,), daemon=True)
    analyzer_thread.start()

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