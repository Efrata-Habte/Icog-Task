import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


def create_figures_dir():
    if not os.path.exists('figures'):
        os.makedirs('figures')


def plot_training_curves(history, save_path='figures/training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(len(history['train_loss']))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train Accuracy', alpha=0.7)
    axes[1].plot(epochs, history['val_acc'], label='Val Accuracy', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='figures/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    i = 0
    while i < cm.shape[0]:
        j = 0
        while j < cm.shape[1]:
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            j = j + 1
        i = i + 1

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def plot_weight_heatmap(weights, layer_name, feature_names=None,
                       save_path='figures/weights_heatmap.png'):
    fig, ax = plt.subplots(figsize=(10, 6))

    limit = float(np.abs(weights).max())
    im = ax.imshow(weights.T, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')

    if feature_names is not None:
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=60, ha='right', fontsize=8)

    ax.set_xlabel('Input features')
    ax.set_ylabel('Neurons')
    ax.set_title(f'Weights: {layer_name}')

    plt.colorbar(im, ax=ax, label='Weight (blue negative, red positive)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def plot_spiral_decision_boundary(model, X, y, epoch, save_path=None):
    h = 0.02
    x_min = X[:, 0].min() - 0.1
    x_max = X[:, 0].max() + 0.1
    y_min = X[:, 1].min() - 0.1
    y_max = X[:, 1].max() + 0.1

    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)
    XX, YY = np.meshgrid(xx, yy)

    Z_input = np.c_[XX.ravel(), YY.ravel()]
    Z_pred = model.forward(Z_input)
    Z = np.argmax(Z_pred, axis=1)
    Z = Z.reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.contourf(XX, YY, Z, alpha=0.3, cmap='viridis')

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                        edgecolors='black', s=50, alpha=0.7)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'Decision Boundary (Epoch {epoch})')

    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_architecture_diagram(layer_sizes, save_path='figures/architecture.png'):
    fig, ax = plt.subplots(figsize=(10, 6))

    max_neurons = max(layer_sizes)
    layer_width = 1.0
    num_layers = len(layer_sizes)

    layer_positions = np.linspace(0, num_layers - 1, num_layers)

    circles = []

    i = 0
    while i < num_layers:
        layer_size = layer_sizes[i]
        x_pos = layer_positions[i]

        if layer_size <= 10:
            neuron_positions = np.linspace(0, 1, layer_size)
            j = 0
            while j < layer_size:
                y_pos = neuron_positions[j]
                circle = plt.Circle((x_pos, y_pos), 0.05, color='blue', alpha=0.7)
                ax.add_patch(circle)
                circles.append((x_pos, y_pos))
                j = j + 1
        else:
            neuron_positions = np.linspace(0, 1, min(5, layer_size))
            j = 0
            while j < len(neuron_positions):
                y_pos = neuron_positions[j]
                circle = plt.Circle((x_pos, y_pos), 0.05, color='blue', alpha=0.7)
                ax.add_patch(circle)
                circles.append((x_pos, y_pos))
                j = j + 1

            ax.text(x_pos + 0.1, 0.5, f"... ({layer_size} total)",
                   fontsize=10, va='center')

        i = i + 1

    i = 0
    while i < len(circles) - 1:
        if i < len(circles) - 1:
            x1, y1 = circles[i]
            j = i + 1
            while j < len(circles):
                if j < len(circles):
                    x2, y2 = circles[j]
                    if abs(x2 - x1 - 1.0) < 0.1:
                        ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.2, linewidth=0.5)
                j = j + 1
        i = i + 1

    ax.set_xlim(-0.5, num_layers - 0.5 + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Network Architecture')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
