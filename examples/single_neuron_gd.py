import numpy as np


def single_neuron_demo():
    np.random.seed(42)

    x_input = 2.0
    w1 = 0.5
    b1 = 0.1

    w2 = -0.3
    b2 = 0.2

    w3 = 0.8
    b3 = -0.1

    y_true = 1

    learning_rate = 0.01

    print("Single Neuron Gradient Descent Demo")
    print("=" * 60)
    print(f"Initial weights: w1={w1}, b1={b1}, w2={w2}, b2={b2}, w3={w3}, b3={b3}")
    print(f"Input: x={x_input}, True output: y={y_true}")
    print("=" * 60)

    step = 0
    while step < 5:
        print(f"\n--- Step {step + 1} ---")

        z1 = x_input * w1 + b1
        a1 = max(0, z1)

        z2 = a1 * w2 + b2
        a2 = max(0, z2)

        z3 = a2 * w3 + b3
        y_pred = z3

        loss = 0.5 * (y_pred - y_true) ** 2

        print(f"Forward pass:")
        print(f"  z1 = {x_input} * {w1} + {b1} = {z1:.4f}")
        print(f"  a1 = ReLU({z1:.4f}) = {a1:.4f}")
        print(f"  z2 = {a1:.4f} * {w2} + {b2} = {z2:.4f}")
        print(f"  a2 = ReLU({z2:.4f}) = {a2:.4f}")
        print(f"  z3 = {a2:.4f} * {w3} + {b3} = {z3:.4f}")
        print(f"  y_pred = {y_pred:.4f}")
        print(f"  loss = 0.5 * ({y_pred:.4f} - {y_true})^2 = {loss:.4f}")

        dz3 = y_pred - y_true
        dw3 = dz3 * a2
        db3 = dz3
        da2 = dz3 * w3

        if z2 > 0:
            dz2 = da2
        else:
            dz2 = 0

        dw2 = dz2 * a1
        db2 = dz2
        da1 = dz2 * w2

        if z1 > 0:
            dz1 = da1
        else:
            dz1 = 0

        dw1 = dz1 * x_input
        db1 = dz1

        print("Backward pass:")
        print(f"  dz3 = {y_pred:.4f} - {y_true} = {dz3:.4f}")
        print(f"  dw3 = {dz3:.4f} * {a2:.4f} = {dw3:.4f}")
        print(f"  db3 = {db3:.4f}")
        print(f"  da2 = {dz3:.4f} * {w3} = {da2:.4f}")
        print(f"  dz2 = {dz2:.4f}")
        print(f"  dw2 = {dz2:.4f} * {a1:.4f} = {dw2:.4f}")
        print(f"  db2 = {db2:.4f}")
        print(f"  da1 = {dz2:.4f} * {w2} = {da1:.4f}")
        print(f"  dz1 = {dz1:.4f}")
        print(f"  dw1 = {dz1:.4f} * {x_input} = {dw1:.4f}")
        print(f"  db1 = {db1:.4f}")

        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        w3 = w3 - learning_rate * dw3
        b3 = b3 - learning_rate * db3

        print(f"Weight updates (lr={learning_rate}):")
        print(f"  w1: {w1 + learning_rate * dw1:.4f} -> {w1:.4f}")
        print(f"  b1: {b1 + learning_rate * db1:.4f} -> {b1:.4f}")
        print(f"  w2: {w2 + learning_rate * dw2:.4f} -> {w2:.4f}")
        print(f"  b2: {b2 + learning_rate * db2:.4f} -> {b2:.4f}")
        print(f"  w3: {w3 + learning_rate * dw3:.4f} -> {w3:.4f}")
        print(f"  b3: {b3 + learning_rate * db3:.4f} -> {b3:.4f}")

        step = step + 1


if __name__ == "__main__":
    single_neuron_demo()
