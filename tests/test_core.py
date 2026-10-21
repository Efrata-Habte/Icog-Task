from pathlib import Path

import numpy as np
import pytest

from drybean.data import load_dry_bean
from drybean.gradient_check import gradient_check
from drybean.losses import Loss_CategoricalCrossentropy
from drybean.neural_network import NeuralNetwork
from drybean.optimizer import Optimizer_Adam
from drybean.train import train_mini_batch
from drybean.weights import load_weights, save_weights

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'Dry_Bean_Dataset.xlsx'


def make_blobs(n_per_class=60, classes=3, features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = np.concatenate([rng.randn(n_per_class, features) + c * 4.0 for c in range(classes)])
    y = np.repeat(np.arange(classes), n_per_class)
    return X, y


def test_softmax_outputs_are_probabilities():
    np.random.seed(0)
    X, _ = make_blobs()
    probs = NeuralNetwork([4, 8, 3]).forward(X)
    assert probs.shape == (X.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_backprop_matches_numerical_gradients():
    np.random.seed(0)
    X, y = make_blobs(n_per_class=10)
    X = X / np.abs(X).max()  # keep softmax away from saturation, where the loss clip hides gradients
    model = NeuralNetwork([4, 8, 3])
    max_error = gradient_check(model, Loss_CategoricalCrossentropy(), X, y)
    assert max_error < 1e-4


def test_training_learns_separable_data():
    np.random.seed(0)
    X, y = make_blobs()
    model = NeuralNetwork([4, 16, 3])
    history = train_mini_batch(model, Optimizer_Adam(learning_rate=0.01), Loss_CategoricalCrossentropy(),
                               X, y, epochs=30, batch_size=32, verbose=False)
    assert history['train_loss'][-1] < history['train_loss'][0]
    assert history['train_acc'][-1] > 0.95
    assert 'val_loss' not in history


def test_history_tracks_optional_splits():
    np.random.seed(0)
    X, y = make_blobs()
    history = train_mini_batch(NeuralNetwork([4, 8, 3]), Optimizer_Adam(), Loss_CategoricalCrossentropy(),
                               X, y, X, y, epochs=2, batch_size=32, X_test=X, y_test=y, verbose=False)
    for key in ('train', 'val', 'test'):
        assert len(history[key + '_loss']) == 2
        assert len(history[key + '_acc']) == 2


def test_weights_round_trip(tmp_path):
    np.random.seed(0)
    X, _ = make_blobs()
    model = NeuralNetwork([4, 8, 3])
    save_weights(model, str(tmp_path / 'w.npz'))
    other = NeuralNetwork([4, 8, 3])
    load_weights(other, str(tmp_path / 'w.npz'))
    assert np.allclose(model.forward(X), other.forward(X))


@pytest.mark.skipif(not DATA_PATH.exists(), reason='Dry Bean dataset not downloaded')
def test_dry_bean_split_is_stratified_and_scaled():
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dry_bean(str(DATA_PATH))
    total = len(y_train) + len(y_val) + len(y_test)
    assert abs(len(y_train) / total - 0.70) < 0.01
    assert abs(len(y_val) / total - 0.15) < 0.01
    assert abs(len(y_test) / total - 0.15) < 0.01
    assert len(class_names) == 7
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-6)
