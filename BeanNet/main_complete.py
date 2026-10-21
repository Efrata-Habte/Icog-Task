import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from losses import Loss_CategoricalCrossentropy
from optimizer import Optimizer_Adam, Optimizer_SGD
from data import create_spiral_data, load_dry_bean
from train import train_mini_batch, evaluate
from visualize import (create_figures_dir, plot_training_curves,
                           plot_confusion_matrix, plot_spiral_decision_boundary,
                           plot_architecture_diagram, plot_weight_heatmap)
from weights import save_weights, get_layer_weights
from predictions import print_predictions_table, per_class_metrics

np.random.seed(42)

print("=" * 60)
print("SPIRAL DATASET TRAINING")
print("=" * 60)

X_spiral, y_spiral = create_spiral_data(samples=100, classes=3)
print(f"Spiral data: X shape {X_spiral.shape}, y shape {y_spiral.shape}")

model_spiral = NeuralNetwork([2, 64, 3])
optimizer_spiral = Optimizer_Adam(learning_rate=0.01)
loss_fn = Loss_CategoricalCrossentropy()

print(f"Model parameters: {model_spiral.count_parameters()}")

history_spiral = train_mini_batch(
    model_spiral, optimizer_spiral, loss_fn,
    X_spiral, y_spiral, X_spiral, y_spiral,
    epochs=1000, batch_size=32
)

create_figures_dir()
plot_training_curves(history_spiral, 'figures/spiral_training_curves.png')
plt.show(block=False)

plot_spiral_decision_boundary(model_spiral, X_spiral, y_spiral, epoch=0,
                             save_path='figures/spiral_boundary_epoch0.png')
plot_spiral_decision_boundary(model_spiral, X_spiral, y_spiral, epoch=500,
                             save_path='figures/spiral_boundary_epoch500.png')
plot_spiral_decision_boundary(model_spiral, X_spiral, y_spiral, epoch=999,
                             save_path='figures/spiral_boundary_final.png')

plot_architecture_diagram([2, 64, 3], 'figures/spiral_architecture.png')

save_weights(model_spiral, 'figures/spiral_weights.npz')

print("\nSpiral training complete!")

print("\n" + "=" * 60)
print("DRY BEAN DATASET TRAINING")
print("=" * 60)

try:
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dry_bean('data/Dry_Bean_Dataset.xlsx')

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {len(class_names)}")

    model_bean = NeuralNetwork([16, 64, 32, 7])
    optimizer_bean = Optimizer_Adam(learning_rate=0.001)

    print(f"Model parameters: {model_bean.count_parameters()}")

    history_bean = train_mini_batch(
        model_bean, optimizer_bean, loss_fn,
        X_train, y_train, X_val, y_val,
        epochs=100, batch_size=64
    )

    test_loss, test_acc, y_pred, y_probs = evaluate(model_bean, loss_fn, X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_training_curves(history_bean, 'figures/bean_training_curves.png')
    plot_confusion_matrix(y_test, y_pred, class_names, 'figures/bean_confusion_matrix.png')
    plot_architecture_diagram([16, 64, 32, 7], 'figures/bean_architecture.png')

    weights_layer0 = get_layer_weights(model_bean, 0)
    if weights_layer0 is not None:
        feature_names = [f"Feature {i}" for i in range(16)]
        plot_weight_heatmap(weights_layer0, "Layer 0 (Input → Hidden)",
                           feature_names=feature_names,
                           save_path='figures/bean_weights_layer0.png')

    print("\nPrediction Examples:")
    print_predictions_table(model_bean, X_test, y_test, class_names, num_samples=15)

    print("\nPer-Class Performance:")
    per_class_metrics(y_test, y_pred, class_names)

    save_weights(model_bean, 'figures/bean_weights.npz')

    print("\nDry Bean training complete!")

except FileNotFoundError:
    print("Dry_Bean_Dataset.xlsx not found. Skipping Dry Bean training.")

print("\n" + "=" * 60)
print("All training and visualizations complete!")
print("Figures saved to figures/ directory")
print("=" * 60)

plt.show(block=True)
