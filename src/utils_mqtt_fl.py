# utils_mqtt_fl.py

import json
import base64
import numpy as np
import tensorflow as tf
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


def quantize_model_tflite(model):
    """
    Converts a Keras model to a quantized TensorFlow Lite model.
    Returns the model as a byte array.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables the standard 8-bit quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # --- DEFINITIVE FIX for the LSTM Error ---
    # This tells the converter to include TensorFlow's core operations (like TensorListReserve)
    # in the final model instead of trying to convert them, which avoids the error.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite ops.
        tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
    ]
    # This flag is also recommended by the error message to prevent the conversion failure.
    converter._experimental_lower_tensor_list_ops = False
    # --- END OF FIX ---

    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

def set_tflite_model_weights(interpreter, weights):
    """
    Sets the weights of a TensorFlow Lite interpreter from a list of numpy arrays.
    NOTE: This is a more advanced function. For our workflow, we will send the
    already-quantized model, so the client won't need to set weights manually.
    This is here for reference.
    """
    for i, tensor_details in enumerate(interpreter.get_tensor_details()):
        if tensor_details['name'] in [w.name for w in model.trainable_weights]:
             # Find the corresponding weight in the list and set it
            interpreter.set_tensor(tensor_details['index'], weights[i])