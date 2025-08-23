# Stage 1: Build the Python environment with dependencies
# This 'builder' stage compiles dependencies and can be discarded later
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt and install Python packages
# This is done early to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---
# Stage 2: Create the final, smaller production image
FROM python:3.12-slim

WORKDIR /app

# Copy the installed Python packages from the builder stage
# COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local /usr/local

# Copy the entrypoint script (we will create this next)
COPY entrypoint.sh .
# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Copy your application code
COPY . .

# Set the entrypoint. This script will run when any container starts.
ENTRYPOINT ["./entrypoint.sh"]