import argparse

import numpy as np

from .data import load_dry_bean
from .losses import Loss_CategoricalCrossentropy
from .neural_network import NeuralNetwork
from .optimizer import Optimizer_Adam, Optimizer_SGD
from .predictions import per_class_metrics
from .train import evaluate, train_mini_batch
from .weights import save_weights


def build_parser():
    parser = argparse.ArgumentParser(prog='drybean-train', description='Train a dry bean classifier.')
    parser.add_argument('--data', default='data/Dry_Bean_Dataset.xlsx', help='path to Dry_Bean_Dataset.xlsx')
    parser.add_argument('--hidden', type=int, nargs='+', default=[64, 32], help='hidden layer sizes')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-weights', help='write the trained weights to this .npz file')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)

    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dry_bean(args.data)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    layer_sizes = [X_train.shape[1]] + args.hidden + [len(class_names)]
    model = NeuralNetwork(layer_sizes)
    print(f"Architecture: {layer_sizes} ({model.count_parameters()} parameters)")

    if args.optimizer == 'adam':
        optimizer = Optimizer_Adam(learning_rate=args.lr)
    else:
        optimizer = Optimizer_SGD(learning_rate=args.lr, momentum=0.9)
    loss_fn = Loss_CategoricalCrossentropy()

    train_mini_batch(model, optimizer, loss_fn, X_train, y_train, X_val, y_val,
                     epochs=args.epochs, batch_size=args.batch_size)

    test_loss, test_acc, y_pred, _ = evaluate(model, loss_fn, X_test, y_test)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    per_class_metrics(y_test, y_pred, class_names)

    if args.save_weights:
        save_weights(model, args.save_weights)
        print(f"Weights saved to {args.save_weights}")


if __name__ == '__main__':
    main()
