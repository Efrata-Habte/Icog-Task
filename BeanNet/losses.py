import numpy as np

class Loss_CategoricalCrossentropy:
    def calculate(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)

        batch_size = y_pred.shape[0]
        log_likelihoods = np.log(y_pred)

        loss_sum = 0.0
        i = 0
        while i < batch_size:
            true_class = int(y_true[i])
            loss_sum = loss_sum + log_likelihoods[i, true_class]
            i = i + 1

        loss = -loss_sum / batch_size
        return loss
