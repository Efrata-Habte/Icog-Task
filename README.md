# BeanNet: A Neural Network From Scratch, Watched Live

A neural network written in plain NumPy, with no deep-learning framework. It learns to classify seven dry bean varieties and to untangle a 3-class spiral, and a dark-mode dashboard lets you watch loss and accuracy change as it trains.

## Project layout

All code is in `BeanNet/`.

| File | Purpose |
|---|---|
| `neural_network.py` | Dense layers, ReLU / Softmax / Sigmoid / Tanh, `NeuralNetwork` |
| `losses.py` | Categorical cross-entropy |
| `optimizer.py` | SGD (with momentum) and Adam |
| `data.py` | Spiral generator and Dry Bean loader (stratified 70/15/15 split) |
| `train.py` | Mini-batch training loop and evaluation |
| `visualize.py`, `weights.py`, `predictions.py` | Plots, weight export, per-class metrics |
| `gradient_check.py` | Numerical vs analytical gradient check |
| `single_neuron_gd.py` | Step-by-step forward/backward demo on a tiny network |
| `experiments_complete.py` | Architecture and optimizer comparisons |
| `main_complete.py` | Full pipeline: trains both datasets, saves figures |
| `dashboard_darkmode.py` | Flask dashboard with live loss, accuracy and charts |

## Setup

```bash
cd BeanNet
pip install -r requirements.txt
pip install flask plotly      # dashboard only
```

Run everything from inside `BeanNet/`, because the dataset path (`data/Dry_Bean_Dataset.xlsx`) is relative.

## Usage

```bash
python main_complete.py         # train on spiral + Dry Bean, save figures
python experiments_complete.py  # compare architectures and optimizers
python gradient_check.py        # verify backprop
python single_neuron_gd.py      # tiny worked example
python dashboard_darkmode.py    # dashboard at http://127.0.0.1:5000
```

## Dashboard

Pick the layers, learning rate and epochs, then click **Train Network**. While it trains you get:
- live cards for epoch, loss, accuracy, and test (and validation) loss and accuracy
- loss and accuracy curves that update every epoch
- the architecture diagram, weight heatmaps and, for the spiral, the decision boundary

## Data and evaluation

- **Dry Bean:** stratified 70% train / 15% validation / 15% test. The scaler is fit on the training split only.
- **Spiral:** 300 training points and a separate freshly generated 300-point test set.

Typical results (Adam, lr 0.01, two hidden layers):

| Dataset | Train acc | Test acc |
|---|---|---|
| Spiral (200 epochs) | ~95% | ~87% |
| Dry Bean (50 epochs) | ~95% | ~92% |

## Design notes

- He initialization, weights shaped `(n_inputs, n_neurons)`, batch dimension first.
- Softmax and cross-entropy are combined, so the logit gradient is `(y_pred - y_one_hot) / batch_size`.
- Mini-batches are reshuffled every epoch.
- The code uses explicit loops and plain NumPy on purpose, so each step is clear to follow.
