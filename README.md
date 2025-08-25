# Adaptive Federated Learning for Context-Aware Intrusion Detection in Heterogeneous IoT Networks

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)
![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete, end-to-end, privacy-preserving Intrusion Detection System (IDS) built on a Federated Learning (FL) architecture. This project demonstrates a fully automated MLOps pipeline, from collaborative model training across distributed clients to real-time deployment and live threat detection with multi-channel alerting.

The entire system is containerized with Docker for seamless setup and scalability.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Core Features](#2-core-features)
3. [System Architecture](#3-system-architecture)
4. [Live System Demo](#4-live-system-demo)
5. [Performance Results](#5-performance-results)
6. [Getting Started](#6-getting-started)
   - [Prerequisites](#prerequisites)
   - [Initial Setup](#initial-setup)
   - [Running with Docker Compose](#running-with-docker-compose)
7. [Directory Structure](#7-directory-structure)
8. [Scripts Overview](#8-scripts-overview)

---

## 1. Project Overview

This project implements an IDS that trains a deep learning model on network traffic data distributed across multiple clients (e.g., IoT devices, Raspberry Pis) without centralizing the raw, private data. The workflow is designed to run automatically on a schedule (e.g., daily), ensuring the detection model is continuously improved and deployed without any manual intervention.

## 2. Core Features

*   **Privacy-Preserving Training:** Uses Federated Learning to train a global model on decentralized data.
*   **Fully Automated MLOps Pipeline:** A "fire-and-forget" system that handles daily retraining, model aggregation, and deployment.
*   **Scalable Pull-Based Deployment:** Clients are notified via MQTT and pull the latest model from a central HTTP server, enabling massive scalability.
*   **Live Intrusion Detection:** Deployed models monitor live network traffic using Scapy to detect and classify threats in real-time.
*   **Stateful Alerting System:** Differentiates between single anomalies and sustained attacks, reducing alert fatigue.
*   **Dual-Channel Notifications:** Issues immediate alerts via both **MQTT** (for automated prevention systems) and **formatted HTML emails** (for human administrators).
*   **Containerized & Portable:** The entire application stack (broker, server, clients) is managed with Docker and Docker Compose for a one-command setup.

## 3. System Architecture

<img width="3840" height="2396" alt="Untitled diagram _ Mermaid Chart-2025-08-25-122709" src="https://github.com/user-attachments/assets/4cb50301-1ff0-4eee-b959-1771c6f580b0" />

1.  **Orchestrator Starts:** Kicks off the daily cycle by launching a temporary HTTP server.
2.  **FL Server Begins:** The federated learning server starts, coordinating the training rounds.
3.  **Clients Train:** Clients receive the global model, train on local data, and send updates back.
4.  **Server Aggregates & Saves:** The server combines updates, finalizes the new model, and saves it to a directory served by the HTTP server.
5.  **MQTT Notification:** The orchestrator publishes a message to an MQTT topic with the download URL for the new model.
6.  **Clients Self-Update:** All clients, listening on the topic, receive the URL and download the new model, completing the deployment.

## 4. Live System Demo

The system was validated in a live test environment.

<img width="1203" height="3839" alt="Untitled diagram _ Mermaid Chart-2025-08-25-123229" src="https://github.com/user-attachments/assets/617171ab-3872-4c2f-8492-07a1644c02bb" />


*   **Test:** A Raspberry Pi client running the `live_detection.py` service was targeted by a UDP flood attack from another machine.
*   **Result:** The system successfully identified the sustained attack and triggered both alerts.
    *   **MQTT Alert:** A JSON payload with threat details was published for machine-to-machine communication.
    *   **Email Alert:** A human-readable HTML email was sent to the administrator.


## 5. Performance Results

Two federated averaging strategies were compared: a **Baseline** (sample-based) and an **Adaptive** (loss-based) model. The Adaptive strategy proved superior in the most critical real-world metric.

| Metric | Baseline F1 | Adaptive F1 | Winner |
| :--- | :--- | :--- | :--- |
| **MACRO AVG F1** | **0.6501** | 0.6421 | Baseline |
| **WEIGHTED AVG F1** | 0.9954 | **0.9956** | **Adaptive** |

The **Weighted F1-Score** is paramount as it accounts for the natural class imbalance of network traffic. The Adaptive model's victory here indicates it is more reliable for production use. It is hypothesized that the Adaptive model's performance will improve further as more clients are added to the federation.

---

## 6. Getting Started

### Prerequisites

*   [Docker](https://www.docker.com/get-started)
*   [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)
*   A preprocessed dataset (`cicids2017_preprocessed.csv`) in the `data/Preprocessed/` directory.

### Initial Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Partition the Data:** Run the data partitioner script to create the non-IID datasets for your clients.
    ```bash
    # You will need a local Python/Conda environment for this one-time setup
    pip install pandas scikit-learn
    python src/data_practioner.py
    ```

3.  **Configure the MQTT Broker:**
    *   Create a directory: `mkdir mosquitto_config`
    *   Create a password file (you will be prompted for a password):
        ```bash
        docker run -it --rm -v "$(pwd)/mosquitto_config:/mosquitto/config" eclipse-mosquitto:2 mosquitto_passwd -c /mosquitto/config/passwordfile your_username
        ```
    *   Create a config file `mosquitto_config/mosquitto.conf` with the following content:
        ```
        allow_anonymous false
        password_file /mosquitto/config/passwordfile
        ```

4.  **Update Credentials:** Open `docker-compose.yml` and update the `MQTT_USERNAME` and `MQTT_PASSWORD` environment variables to match what you just created.

### Running with Docker Compose

1.  **Build the Docker Images:**
    ```bash
    docker-compose build
    ```

2.  **Start the Core Services:** This launches the MQTT broker and the FL server in the background.
    ```bash
    docker-compose up -d broker server
    ```

3.  **Launch the Clients:** Use the launcher script to start any number of clients. For example, to start 4 clients:
    ```bash
    chmod +x start_clients.sh
    ./start_clients.sh 4
    ```
    The training process will now begin.

4.  **Monitor the System:** View the logs of all running containers.
    ```bash
    docker-compose logs -f
    ```

5.  **Shut Down:** Stop and remove all containers.
    ```bash
    docker-compose down
    ```

---

## 7. Directory Structure
```
├── data/
│ ├── Preprocessed/ # Main dataset
│ └── federated_data/ # Client data partitions
├── saved_models/ # Saved global models
├── src/
│ ├── server_federated_mqtt.py
│ ├── client_federated_mqtt.py
│ ├── live_detection.py
│ └── ... (other scripts)
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── start_clients.sh
├── requirements.txt
└── README.md
```

## 8. Scripts Overview

*   **`server_federated_mqtt.py`**: The FL aggregation server.
*   **`client_federated_mqtt.py`**: The FL training client.
*   **`live_detection.py`**: The real-time IDS script deployed on clients.
*   **`model_updater.py`**: The background service on clients that listens for and downloads new models.
*   **`train_and_deploy.sh`**: The master orchestration script (used for non-Docker testing).
*   **`data_practioner.py`**: Splits the main dataset into non-IID client partitions.
*   **`generate_report.py`**: Evaluates models and creates performance comparison plots.
*   **`attack_raspberrypi.py`**: A simple UDP flood script for testing the live IDS.
*   **`utils_mqtt_fl.py`**: Shared helper functions for serialization and aggregation.
