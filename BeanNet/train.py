import numpy as np


def train_mini_batch(model, optimizer, loss_fn, X_train, y_train, X_val, y_val,
                     epochs, batch_size):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    epoch = 0
    while epoch < epochs:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        batch_start = 0
        while batch_start < X_train.shape[0]:
            batch_end = min(batch_start + batch_size, X_train.shape[0])
            batch_indices = indices[batch_start:batch_end]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            y_pred = model.forward(X_batch)
            model.backward(y_batch)
            optimizer.update_params(model.layers)

            batch_start = batch_end

        y_pred_train = model.forward(X_train)
        train_loss = loss_fn.calculate(y_pred_train, y_train)
        train_acc = np.mean(np.argmax(y_pred_train, axis=1) == y_train)

        y_pred_val = model.forward(X_val)
        val_loss = loss_fn.calculate(y_pred_val, y_val)
        val_acc = np.mean(np.argmax(y_pred_val, axis=1) == y_val)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        epoch = epoch + 1

    return history


def evaluate(model, loss_fn, X_test, y_test):
    y_pred = model.forward(X_test)
    test_loss = loss_fn.calculate(y_pred, y_test)
    test_acc = np.mean(np.argmax(y_pred, axis=1) == y_test)

    pred_classes = np.argmax(y_pred, axis=1)

    return test_loss, test_acc, pred_classes, y_pred
