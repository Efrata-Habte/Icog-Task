import numpy as np
import time
from drybean.neural_network import NeuralNetwork
from drybean.losses import Loss_CategoricalCrossentropy
from drybean.optimizer import Optimizer_Adam, Optimizer_SGD
from drybean.data import load_dry_bean
from examples.spiral import create_spiral_data
from drybean.train import train_mini_batch, evaluate
from dashboard.visualize import plot_training_curves
from pathlib import Path

DATA_PATH = str(Path(__file__).resolve().parent.parent / 'data' / 'Dry_Bean_Dataset.xlsx')

np.random.seed(42)


def experiment_architectures_spiral():
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON ON SPIRAL")
    print("=" * 80)

    X, y = create_spiral_data(samples=100, classes=3)

    architectures = [
        [2, 16, 3],
        [2, 32, 16, 3],
        [2, 64, 32, 16, 3]
    ]

    loss_fn = Loss_CategoricalCrossentropy()

    results = []

    i = 0
    while i < len(architectures):
        arch = architectures[i]

        model = NeuralNetwork(arch)
        optimizer = Optimizer_Adam(learning_rate=0.01)

        param_count = model.count_parameters()

        start_time = time.time()
        history = train_mini_batch(
            model, optimizer, loss_fn,
            X, y, X, y,
            epochs=1000, batch_size=32
        )
        train_time = time.time() - start_time

        test_loss, test_acc, _, _ = evaluate(model, loss_fn, X, y)
        train_acc = history['train_acc'][-1]

        results.append({
            'architecture': str(arch),
            'params': param_count,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time
        })

        print(f"\nArchitecture: {arch}")
        print(f"  Parameters: {param_count}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Time: {train_time:.2f}s")

        i = i + 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    j = 0
    while j < len(results):
        r = results[j]
        print(f"{r['architecture']:<20} | Params: {r['params']:>6} | "
              f"Train: {r['train_acc']:.4f} | Test: {r['test_acc']:.4f} | "
              f"Time: {r['time']:.2f}s")
        j = j + 1


def experiment_architectures_dry_bean():
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON ON DRY BEAN")
    print("=" * 80)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dry_bean(DATA_PATH)

        architectures = [
            [16, 16, 7],
            [16, 32, 16, 7],
            [16, 64, 32, 16, 7]
        ]

        loss_fn = Loss_CategoricalCrossentropy()

        results = []

        i = 0
        while i < len(architectures):
            arch = architectures[i]

            model = NeuralNetwork(arch)
            optimizer = Optimizer_Adam(learning_rate=0.001)

            param_count = model.count_parameters()

            start_time = time.time()
            history = train_mini_batch(
                model, optimizer, loss_fn,
                X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64
            )
            train_time = time.time() - start_time

            test_loss, test_acc, _, _ = evaluate(model, loss_fn, X_test, y_test)
            train_acc = history['train_acc'][-1]

            results.append({
                'architecture': str(arch),
                'params': param_count,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'time': train_time
            })

            print(f"\nArchitecture: {arch}")
            print(f"  Parameters: {param_count}")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Time: {train_time:.2f}s")

            i = i + 1

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        j = 0
        while j < len(results):
            r = results[j]
            print(f"{r['architecture']:<20} | Params: {r['params']:>6} | "
                  f"Train: {r['train_acc']:.4f} | Test: {r['test_acc']:.4f} | "
                  f"Time: {r['time']:.2f}s")
            j = j + 1

    except FileNotFoundError:
        print("Dry_Bean_Dataset.xlsx not found. Skipping.")


def experiment_optimizers():
    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON ON SPIRAL")
    print("=" * 80)

    X, y = create_spiral_data(samples=100, classes=3)
    loss_fn = Loss_CategoricalCrossentropy()

    optimizers_list = [
        ('SGD (lr=1.0)', Optimizer_SGD(learning_rate=1.0, momentum=0.0)),
        ('SGD+Momentum (lr=0.1)', Optimizer_SGD(learning_rate=0.1, momentum=0.9)),
        ('Adam (lr=0.01)', Optimizer_Adam(learning_rate=0.01))
    ]

    results = []

    i = 0
    while i < len(optimizers_list):
        opt_name, optimizer = optimizers_list[i]

        model = NeuralNetwork([2, 64, 3])

        history = train_mini_batch(
            model, optimizer, loss_fn,
            X, y, X, y,
            epochs=1000, batch_size=32
        )

        final_loss = history['train_loss'][-1]
        final_acc = history['train_acc'][-1]

        results.append({
            'optimizer': opt_name,
            'final_loss': final_loss,
            'final_acc': final_acc,
            'history': history
        })

        print(f"\n{opt_name}:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Final Accuracy: {final_acc:.4f}")

        i = i + 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    j = 0
    while j < len(results):
        r = results[j]
        print(f"{r['optimizer']:<25} | Loss: {r['final_loss']:.4f} | Acc: {r['final_acc']:.4f}")
        j = j + 1

    return results


def experiment_learning_rates():
    print("\n" + "=" * 80)
    print("LEARNING RATE COMPARISON ON SPIRAL")
    print("=" * 80)

    X, y = create_spiral_data(samples=100, classes=3)
    loss_fn = Loss_CategoricalCrossentropy()

    learning_rates = [0.001, 0.01, 0.1]

    results = []

    i = 0
    while i < len(learning_rates):
        lr = learning_rates[i]

        model = NeuralNetwork([2, 64, 3])
        optimizer = Optimizer_Adam(learning_rate=lr)

        history = train_mini_batch(
            model, optimizer, loss_fn,
            X, y, X, y,
            epochs=1000, batch_size=32
        )

        final_loss = history['train_loss'][-1]
        final_acc = history['train_acc'][-1]

        results.append({
            'lr': lr,
            'final_loss': final_loss,
            'final_acc': final_acc
        })

        print(f"\nLearning Rate: {lr}")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Final Accuracy: {final_acc:.4f}")

        i = i + 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    j = 0
    while j < len(results):
        r = results[j]
        print(f"LR: {r['lr']:<10} | Loss: {r['final_loss']:.4f} | Acc: {r['final_acc']:.4f}")
        j = j + 1


if __name__ == "__main__":
    experiment_architectures_spiral()
    experiment_architectures_dry_bean()
    experiment_optimizers()
    experiment_learning_rates()
