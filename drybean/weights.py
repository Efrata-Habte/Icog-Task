import numpy as np


def save_weights(model, filepath):
    weights_dict = {}
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if hasattr(layer, 'weights'):
            layer_name = f"layer_{i}_weights"
            weights_dict[layer_name] = layer.weights
            layer_name_b = f"layer_{i}_biases"
            weights_dict[layer_name_b] = layer.biases
        i = i + 1

    np.savez(filepath, **weights_dict)


def load_weights(model, filepath):
    data = np.load(filepath)
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if hasattr(layer, 'weights'):
            layer_name = f"layer_{i}_weights"
            layer_name_b = f"layer_{i}_biases"
            if layer_name in data:
                layer.weights = data[layer_name]
                layer.biases = data[layer_name_b]
        i = i + 1


def get_layer_weights(model, layer_idx):
    i = 0
    dense_count = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if hasattr(layer, 'weights'):
            if dense_count == layer_idx:
                return layer.weights
            dense_count = dense_count + 1
        i = i + 1
    return None
