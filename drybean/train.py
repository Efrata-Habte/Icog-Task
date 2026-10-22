import numpy as np


def _score(model, loss_fn, X, y):
    y_pred = model.forward(X)
    loss = loss_fn.calculate(y_pred, y)
    acc = np.mean(np.argmax(y_pred, axis=1) == y)
    return float(loss), float(acc)


def _snapshot(model):
    return [(layer.weights.copy(), layer.biases.copy()) for layer in model.layers if hasattr(layer, 'weights')]


def _restore(model, snapshot):
    dense_layers = [layer for layer in model.layers if hasattr(layer, 'weights')]
    for layer, (weights, biases) in zip(dense_layers, snapshot):
        layer.weights = weights.copy()
        layer.biases = biases.copy()


def train_mini_batch(model, optimizer, loss_fn, X_train, y_train, X_val=None, y_val=None,
                     epochs=100, batch_size=64, verbose=True, history=None, patience=None):
    """Train with mini-batches and return a history dict.

    train_* keys are always recorded; val_* keys only when a validation set is given.
    The test set is deliberately not scored here: call `evaluate` once after training,
    so the test score is never used to pick epochs or settings.
    Early stopping: with `patience` set (and a validation set), training stops once
    val_loss has not reached a new minimum for that many epochs, and the weights from the
    best epoch are restored. Without it, the model keeps the weights of the last epoch.

    Pass an existing `history` dict to have it filled in place after every epoch, so
    another thread can read it while training runs (the dashboard does this for its
    live numbers).
    """
    if history is None:
        history = {}
    splits = ['train']
    if X_val is not None:
        splits.append('val')
    for name in splits:
        history.setdefault(name + '_loss', [])
        history.setdefault(name + '_acc', [])

    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    epoch = 0
    while epoch < epochs:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        batch_start = 0
        while batch_start < X_train.shape[0]:
            batch_end = min(batch_start + batch_size, X_train.shape[0])
            batch_indices = indices[batch_start:batch_end]

            model.forward(X_train[batch_indices])
            model.backward(y_train[batch_indices])
            optimizer.update_params(model.layers)

            batch_start = batch_end

        train_loss, train_acc = _score(model, loss_fn, X_train, y_train)
        scores = {'train': (train_loss, train_acc)}
        if X_val is not None:
            scores['val'] = _score(model, loss_fn, X_val, y_val)

        for name, (loss, acc) in scores.items():
            history[name + '_loss'].append(loss)
            history[name + '_acc'].append(acc)

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            parts = [f"{name}_loss={loss:.4f}, {name}_acc={acc:.4f}" for name, (loss, acc) in scores.items()]
            print(f"Epoch {epoch}: " + ", ".join(parts))

        if patience is not None and X_val is not None:
            val_loss = scores['val'][0]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = _snapshot(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement + 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}: no val_loss improvement for {patience} epochs")
                    break

        epoch = epoch + 1

    if best_weights is not None:
        _restore(model, best_weights)

    return history


def evaluate(model, loss_fn, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = loss_fn.calculate(y_pred, y_test)
    test_acc = np.mean(np.argmax(y_pred, axis=1) == y_test)

    pred_classes = np.argmax(y_pred, axis=1)

    return test_loss, test_acc, pred_classes, y_pred
