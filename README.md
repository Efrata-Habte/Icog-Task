# drybean-classifier

A neural network written in plain NumPy, with no deep-learning framework, that classifies seven dry bean varieties from 16 shape measurements. A separate dark-mode dashboard lets you watch loss and accuracy change as it trains.

## Layout

```
drybean/       core package: model, losses, optimizers, data, training, metrics, CLI
dashboard/     Flask dashboard and plotting (imports drybean; the core never imports it)
examples/      single-neuron walkthrough, spiral sanity check, experiments, plotted training run
tests/         gradient check and unit tests
data/          Dry_Bean_Dataset.xlsx (not tracked)
```

The core depends only on numpy, pandas, scikit-learn and openpyxl. Plotting and web code live outside it.

## Setup

```bash
pip install -r requirements.txt   # or: pip install -e ".[dashboard,dev]"
```

Run everything from the repository root, because the data path is relative.

## Usage

```bash
python -m drybean.cli --epochs 100          # train and evaluate on Dry Bean
python -m drybean.cli --hidden 64 49 --lr 0.01 --save-weights weights.npz
python -m dashboard.app                     # live dashboard at http://127.0.0.1:5000
python -m pytest                            # gradient check and unit tests
```

With `pip install -e .` the training command is also available as `drybean-train`.

Examples:

```bash
python -m examples.single_neuron_gd     # step-by-step forward/backward on a tiny network
python -m examples.experiments          # architecture and optimizer comparisons
python -m examples.train_with_plots     # spiral + Dry Bean runs with saved figures
```

## Data and evaluation

Dry Bean is split with stratification into 70% train, 15% validation and 15% test, and the scaler is fit on the training split only. The dashboard shows live train, validation and test loss and accuracy.

The spiral (`examples/spiral.py`) is a synthetic 3-class problem used as a sanity check that the network can learn non-linear boundaries. It is not part of the classifier.

Typical results (Adam, lr 0.01, two hidden layers):

| Dataset | Train acc | Test acc |
|---|---|---|
| Dry Bean (50 epochs) | ~95% | ~92% |
| Spiral (200 epochs) | ~95% | ~87% |

## Design notes

- He initialization, weights shaped `(n_inputs, n_neurons)`, batch dimension first.
- Softmax and cross-entropy are combined, so the logit gradient is `(y_pred - y_one_hot) / batch_size`.
- Mini-batches are reshuffled every epoch.
- Plain loops and NumPy on purpose, so each step is easy to follow.
