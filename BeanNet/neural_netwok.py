import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1.0 - self.output)


class Activation_Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues * (1.0 - self.output * self.output)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        i = 0
        while i < len(layer_sizes) - 1:
            self.layers.append(Layer_Dense(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(Activation_ReLU())
            i = i + 1
        self.output_activation = Activation_Softmax()

    def forward(self, X):
        out = X
        i = 0
        while i < len(self.layers):
            self.layers[i].forward(out)
            out = self.layers[i].output
            i = i + 1
        self.output_activation.forward(out)
        return self.output_activation.output

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        dvalues = self.output_activation.output.copy()
        i = 0
        while i < batch_size:
            true_class = y_true[i]
            dvalues[i, true_class] = dvalues[i, true_class] - 1.0
            i = i + 1
        dvalues = dvalues / batch_size

        i = len(self.layers) - 1
        while i >= 0:
            self.layers[i].backward(dvalues)
            dvalues = self.layers[i].dinputs
            i = i - 1

    def count_parameters(self):
        total = 0
        i = 0
        while i < len(self.layers):
            layer = self.layers[i]
            if isinstance(layer, Layer_Dense):
                total = total + layer.weights.size + layer.biases.size
            i = i + 1
        return total
