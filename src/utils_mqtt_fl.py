# utils_mqtt_fl.py

import json
import base64
import numpy as np
# No pickle needed in this version

def serialize_weights(weights):
    """
    Serializes a list of NumPy arrays (model weights) into a text-safe JSON string
    using base64 encoding for the binary data.
    """
    ser = [{"shape": w.shape, "dtype": str(w.dtype), "data": base64.b64encode(w.tobytes()).decode("utf-8")} for w in weights]
    return json.dumps({"weights": ser})

def deserialize_weights(payload):
    """
    Deserializes a JSON string created by serialize_weights back into a list
    of NumPy arrays.
    """
    obj = json.loads(payload)
    return [np.frombuffer(base64.b64decode(item["data"]), dtype=np.dtype(item["dtype"])).reshape(item["shape"]) for item in obj["weights"]]

def serialize_message(obj):
    """Serializes a simple dictionary to a JSON string."""
    return json.dumps(obj)

def deserialize_message(payload):
    """Deserializes a JSON string to a dictionary."""
    return json.loads(payload)

def weighted_average_adaptive(updates):
    """Performs weighted averaging on the received updates."""
    if not updates: return None
    total_weight = 0; new_weights = None
    for u in updates:
        num_samples = u.get("num_samples", 0)
        if num_samples == 0: continue
        weight = float(num_samples); total_weight += weight
        if new_weights is None:
            new_weights = [w.astype(np.float32) * weight for w in u["weights"]]
        else:
            for i, w in enumerate(u["weights"]):
                new_weights[i] += w.astype(np.float32) * weight
    if total_weight == 0: return None
    return [w / total_weight for w in new_weights]


def adaptive_federated_averaging(updates):
    """
    Performs federated averaging with weights inversely proportional to client loss.
    """
    if not updates:
        return None

    print("[SERVER] Using Adaptive Federated Averaging (Loss-based)...")
    
    # --- Adaptive Weighting Logic ---
    # A small epsilon to prevent division by zero if loss is 0.0
    epsilon = 1e-6 
    
    # Step 1: Calculate the inverse of each client's loss.
    # We add epsilon for numerical stability. Lower loss -> higher inverse value.
    inverse_losses = [1.0 / (u.get("loss", 1.0) + epsilon) for u in updates]
    
    # Step 2: Normalize these inverse losses so they sum to 1.0, creating our weights.
    total_inverse_loss = sum(inverse_losses)
    adaptive_weights = [il / total_inverse_loss for il in inverse_losses]

    print(f"[SERVER] Client losses: {[round(u.get('loss', 1.0), 4) for u in updates]}")
    print(f"[SERVER] Calculated adaptive weights: {[round(w, 4) for w in adaptive_weights]}")
    
    # --- Standard Weighted Averaging using the new weights ---
    new_weights = None
    for i, u in enumerate(updates):
        client_weight = adaptive_weights[i]
        
        if new_weights is None:
            new_weights = [w.astype(np.float32) * client_weight for w in u["weights"]]
        else:
            for j, w in enumerate(u["weights"]):
                new_weights[j] += w.astype(np.float32) * client_weight
    
    return new_weights