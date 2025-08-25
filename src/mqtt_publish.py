#!/usr/bin/env python3
"""
MQTT Publisher Script

Publishes a message to a specified MQTT topic using the paho-mqtt library.
Call this script from the command line with the required arguments.

Usage:
    python mqtt_publish.py <broker_host> <topic> <message> <username> <password>

Example:
    python mqtt_publish.py 192.168.0.250 afl/global/model_ready "Download ready" user pass
"""

import sys
import paho.mqtt.publish as publish


def main():
    """
    Parse command-line arguments and publish a single MQTT message.
    """
    if len(sys.argv) != 6:
        print(
            "Usage: python mqtt_publish.py <broker_host> <topic> <message> <username> <password>"
        )
        sys.exit(1)

    broker_host = sys.argv[1]
    topic = sys.argv[2]
    message = sys.argv[3]
    username = sys.argv[4]
    password = sys.argv[5]    

    try:
        publish.single(
            topic=topic,
            payload=message,
            hostname=broker_host,
            auth={"username": username, "password": password}
        )
        print(f"Message published to topic '{topic}' on broker '{broker_host}'.")
    except Exception as err:
        print(f"Failed to publish message: {err}")
        sys.exit(2)


if __name__ == "__main__":
    main()
