import numpy as np

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = None

    def update_params(self, layers):
        if self.velocities is None:
            self.velocities = []
            i = 0
            while i < len(layers):
                layer = layers[i]
                if hasattr(layer, 'dweights'):
                    self.velocities.append({'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)})
                else:
                    self.velocities.append(None)
                i = i + 1

        i = 0
        while i < len(layers):
            layer = layers[i]
            if hasattr(layer, 'dweights'):
                v_w = self.velocities[i]['w']
                v_b = self.velocities[i]['b']

                v_w = self.momentum * v_w - self.learning_rate * layer.dweights
                v_b = self.momentum * v_b - self.learning_rate * layer.dbiases

                layer.weights = layer.weights + v_w
                layer.biases = layer.biases + v_b

                self.velocities[i]['w'] = v_w
                self.velocities[i]['b'] = v_b
            i = i + 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update_params(self, layers):
        self.t = self.t + 1

        if self.m is None:
            self.m = []
            self.v = []
            i = 0
            while i < len(layers):
                layer = layers[i]
                if hasattr(layer, 'dweights'):
                    self.m.append({'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)})
                    self.v.append({'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)})
                else:
                    self.m.append(None)
                    self.v.append(None)
                i = i + 1

        i = 0
        while i < len(layers):
            layer = layers[i]
            if hasattr(layer, 'dweights'):
                m_w = self.m[i]['w']
                m_b = self.m[i]['b']
                v_w = self.v[i]['w']
                v_b = self.v[i]['b']

                m_w = self.beta1 * m_w + (1.0 - self.beta1) * layer.dweights
                m_b = self.beta1 * m_b + (1.0 - self.beta1) * layer.dbiases
                v_w = self.beta2 * v_w + (1.0 - self.beta2) * (layer.dweights * layer.dweights)
                v_b = self.beta2 * v_b + (1.0 - self.beta2) * (layer.dbiases * layer.dbiases)

                m_w_hat = m_w / (1.0 - self.beta1 ** self.t)
                m_b_hat = m_b / (1.0 - self.beta1 ** self.t)
                v_w_hat = v_w / (1.0 - self.beta2 ** self.t)
                v_b_hat = v_b / (1.0 - self.beta2 ** self.t)

                layer.weights = layer.weights - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                layer.biases = layer.biases - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

                self.m[i]['w'] = m_w
                self.m[i]['b'] = m_b
                self.v[i]['w'] = v_w
                self.v[i]['b'] = v_b
            i = i + 1
