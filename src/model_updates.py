# src/model_updater.py
import os
import paho.mqtt.client as mqtt
import subprocess
import socket

# --- Configuration ---
BROKER_HOST: str = os.environ.get("MQTT_HOST", "192.168.0.250")
BROKER_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME: str = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD: str = os.environ.get("MQTT_PASSWORD", "")
MQTT_TOPIC = "afl/global/model_ready"
CLIENT_ID = f"model_updater_{socket.gethostname()}"

# The destination path for the downloaded model file on this client.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../saved_models")
DESTINATION_PATH = os.path.join(PROJECT_ROOT, "final_global_model_adaptive.weights.h5")

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for successful MQTT connection."""
    if rc == 0:
        print(f"[{CLIENT_ID}] Connected to MQTT Broker.")
        client.subscribe(MQTT_TOPIC)
        print(f"[{CLIENT_ID}] Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"[{CLIENT_ID}] Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback to handle the model update notification."""
    try:
        download_url = msg.payload.decode("utf-8")
        print(f"\n[{CLIENT_ID}] Received new model notification!")
        print(f"[{CLIENT_ID}] Download URL: {download_url}")
        
        # Use wget to download the file. '-O' specifies the output file path.
        # This command will automatically overwrite the old model file.
        command = ["wget", download_url, "-O", DESTINATION_PATH]
        
        print(f"[{CLIENT_ID}] Starting download to: {DESTINATION_PATH}")
        
        # Execute the download command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[{CLIENT_ID}] SUCCESS: New model downloaded and updated.")
        else:
            print(f"[{CLIENT_ID}] ERROR: Failed to download the new model.")
            print(f"[{CLIENT_ID}] Wget stderr: {result.stderr}")
            
    except Exception as e:
        print(f"[{CLIENT_ID}] An error occurred: {e}")

def main():
    """Main function to start the updater client."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID, clean_session=True)
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"[{CLIENT_ID}] Model updater service starting...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    
    # Loop forever to keep listening for the notification message
    client.loop_forever()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{CLIENT_ID}] Shutting down.")