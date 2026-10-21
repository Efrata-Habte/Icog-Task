import numpy as np


def gradient_check(model, loss_fn, X, y, h=1e-5):
    y_pred = model.forward(X)
    loss_orig = loss_fn.calculate(y_pred, y)
    model.backward(y)

    print("\nGradient Check:")
    print("=" * 60)

    layer_idx = 0
    while layer_idx < len(model.layers):
        layer = model.layers[layer_idx]

        if not hasattr(layer, 'dweights'):
            layer_idx = layer_idx + 1
            continue

        print(f"\nLayer {layer_idx} ({type(layer).__name__}):")

        errors = []

        i = 0
        while i < min(layer.weights.shape[0], 3):
            j = 0
            while j < min(layer.weights.shape[1], 3):
                old_weight = layer.weights[i, j]

                layer.weights[i, j] = old_weight + h
                y_pred_plus = model.forward(X)
                loss_plus = loss_fn.calculate(y_pred_plus, y)

                layer.weights[i, j] = old_weight - h
                y_pred_minus = model.forward(X)
                loss_minus = loss_fn.calculate(y_pred_minus, y)

                layer.weights[i, j] = old_weight

                numerical_grad = (loss_plus - loss_minus) / (2.0 * h)
                analytical_grad = layer.dweights[i, j]

                if abs(analytical_grad) + abs(numerical_grad) < 1e-12:
                    relative_error = 0.0
                else:
                    relative_error = abs(analytical_grad - numerical_grad) / max(abs(analytical_grad) + abs(numerical_grad), 1e-12)

                errors.append(relative_error)

                j = j + 1
            i = i + 1

        max_error = max(errors) if errors else 0
        avg_error = sum(errors) / len(errors) if errors else 0

        print(f"  Weights: max_error={max_error:.2e}, avg_error={avg_error:.2e}")

        if max_error > 1e-4:
            print(f"  WARNING: Large gradient error detected!")

        layer_idx = layer_idx + 1

    print("\nGradient check complete.")


if __name__ == "__main__":
    import numpy as np
    from neural_network import NeuralNetwork
    from losses import Loss_CategoricalCrossentropy
    from data import create_spiral_data

    np.random.seed(42)

    X, y = create_spiral_data(samples=10, classes=3)
    model = NeuralNetwork([2, 8, 3])
    loss_fn = Loss_CategoricalCrossentropy()

    gradient_check(model, loss_fn, X, y, h=1e-5)
