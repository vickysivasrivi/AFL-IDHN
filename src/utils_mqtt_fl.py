# utils_mqtt_fl.py

import json
import base64
import numpy as np
from typing import Dict, Any, List, Optional

# --- Serialization Functions ---
def serialize_weights(weights: List[np.ndarray]) -> str:
    """Serializes a list of NumPy arrays (model weights) into a text-safe JSON string.

    Args:
        weights: A list of NumPy arrays from model.get_weights().

    Returns:
        A JSON string containing the serialized weights.
    """
    ser_weights = [
        {
            "shape": w.shape,
            "dtype": str(w.dtype),
            "data": base64.b64encode(w.tobytes()).decode("utf-8"),
        }
        for w in weights
    ]
    return json.dumps({"weights": ser_weights})


def deserialize_weights(payload: str) -> List[np.ndarray]:
    """Deserializes a JSON string back into a list of NumPy arrays.

    Args:
        payload: A JSON string containing the serialized weights.

    Returns:
        A list of NumPy arrays for model.set_weights().
    """
    obj = json.loads(payload)
    return [
        np.frombuffer(
            base64.b64decode(item["data"]), dtype=np.dtype(item["dtype"])
        ).reshape(item["shape"])
        for item in obj["weights"]
    ]


def serialize_message(obj: Dict[str, Any]) -> str:
    """Serializes a simple dictionary to a JSON string."""
    return json.dumps(obj)


def deserialize_message(payload: str) -> Dict[str, Any]:
    """Deserializes a JSON string to a dictionary."""
    return json.loads(payload)


# --- Aggregation Functions ---
def adaptive_federated_averaging(
    updates: List[Dict[str, Any]]
) -> Optional[List[np.ndarray]]:
    """Performs adaptive federated averaging based on client loss.

    Gives a higher weight to clients that report a lower training loss.

    Args:
        updates: A list of update dictionaries from clients.

    Returns:
        A list of NumPy arrays representing the new global model weights.
    """
    if not updates:
        return None

    print("[SERVER] Using Adaptive Federated Averaging (Loss-based)...")
    epsilon = 1e-8

    # Calculate weights inversely proportional to loss
    inverse_losses = [1.0 / (u.get("loss", 1.0) + epsilon) for u in updates]
    total_inverse_loss = sum(inverse_losses)
    adaptive_weights = [il / total_inverse_loss for il in inverse_losses]

    print(f"[SERVER] Client losses: {[round(u.get('loss', 1.0), 4) for u in updates]}")
    print(f"[SERVER] Calculated adaptive weights: {[round(w, 4) for w in adaptive_weights]}")

    # Perform the weighted average of the weights
    new_weights = None
    for i, u in enumerate(updates):
        client_weight = adaptive_weights[i]
        client_weights = u["weights"]
        if new_weights is None:
            new_weights = [layer * client_weight for layer in client_weights]
        else:
            for j, layer in enumerate(client_weights):
                new_weights[j] += layer * client_weight

    return new_weights

def weighted_average_adaptive(
    updates: List[Dict[str, Any]]
) -> Optional[List[np.ndarray]]:
    """Performs simple federated averaging based on sample size.

    This is the baseline (non-adaptive) aggregation method.

    Args:
        updates: A list of update dictionaries from clients.

    Returns:
        A list of NumPy arrays representing the new global model weights.
    """
    if not updates:
        return None

    print("[SERVER - BASELINE] Using Simple Federated Averaging (Sample-based)...")

    total_samples = sum(u.get("num_samples", 0) for u in updates)
    if total_samples == 0:
        return None  # Avoid division by zero

    new_weights = None
    for index, update_dict in enumerate(updates):
        num_samples = update_dict.get("num_samples", 0)
        weight = float(num_samples) / float(total_samples)
        
        client_weights = update_dict["weights"]

        if new_weights is None:
            new_weights = [layer * weight for layer in client_weights]
        else:
            for j, layer in enumerate(client_weights):
                new_weights[j] += layer * weight

    return new_weights