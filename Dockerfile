# # Use Python 3.10 (TensorFlow + TFF work best here)
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libffi-dev \
#     libssl-dev \
#     gcc \
#     g++ \
#     make \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip
# RUN pip install --upgrade pip

# # Copy requirements first (for caching layers)
# COPY requirements.txt /app/

# # Install Python dependencies
# RUN pip install --upgrade pip setuptools wheel \
#  && pip install --no-cache-dir --default-timeout=100 -i https://pypi.org/simple -r requirements.txt

# # Copy project files into container
# COPY . /app

# # Default command (will run federated server unless overridden)
# CMD ["python", "src/mqtt_federated_server.py"]

# Stage 1: Build the Python environment with dependencies
# Using Python 3.10 as specified in your file
FROM python:3.10-slim as builder

# Set working directory inside the container
WORKDIR /app

# Ensure non-interactive mode for apt-get to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed to build some Python packages
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt first for effective Docker caching
COPY requirements.txt .

# Install Python dependencies using a different PyPI mirror for resilience
# This makes the build more resilient to unstable networks
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# Stage 2: Create the final, smaller image with the application code
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy the environment from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the project files
COPY src/ ./src/
COPY data/ ./data/
# COPY certs/ ./certs/

# Default command to run the server script
CMD ["python", "src/f1_server.py"]