import numpy as np


def _score(model, loss_fn, X, y):
    y_pred = model.forward(X)
    loss = loss_fn.calculate(y_pred, y)
    acc = np.mean(np.argmax(y_pred, axis=1) == y)
    return float(loss), float(acc)


def train_mini_batch(model, optimizer, loss_fn, X_train, y_train, X_val=None, y_val=None,
                     epochs=100, batch_size=64, X_test=None, y_test=None, verbose=True, history=None):
    """Train with mini-batches and return a history dict.

    train_* keys are always recorded; val_* and test_* keys only when that split is given.
    Pass an existing `history` dict to have it filled in place after every epoch, so
    another thread can read it while training runs (the dashboard does this for its
    live numbers).
    """
    if history is None:
        history = {}
    splits = ['train']
    if X_val is not None:
        splits.append('val')
    if X_test is not None:
        splits.append('test')
    for name in splits:
        history.setdefault(name + '_loss', [])
        history.setdefault(name + '_acc', [])

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
        if X_test is not None:
            scores['test'] = _score(model, loss_fn, X_test, y_test)

        for name, (loss, acc) in scores.items():
            history[name + '_loss'].append(loss)
            history[name + '_acc'].append(acc)

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            parts = [f"{name}_loss={loss:.4f}, {name}_acc={acc:.4f}" for name, (loss, acc) in scores.items()]
            print(f"Epoch {epoch}: " + ", ".join(parts))

        epoch = epoch + 1

    return history


def evaluate(model, loss_fn, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = loss_fn.calculate(y_pred, y_test)
    test_acc = np.mean(np.argmax(y_pred, axis=1) == y_test)

    pred_classes = np.argmax(y_pred, axis=1)

    return test_loss, test_acc, pred_classes, y_pred
