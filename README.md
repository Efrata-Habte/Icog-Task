# drybean-classifier

A neural network written in plain NumPy, with no deep-learning framework, that classifies seven dry bean varieties from 16 shape measurements. A separate dark-mode dashboard lets you watch loss and accuracy change as it trains.

## Layout

```
drybean/       core package: model, losses, optimizers, data, training, metrics, CLI
dashboard/     Flask dashboard and plotting (imports drybean; the core never imports it)
examples/      single-neuron walkthrough, spiral sanity check, experiments, plotted training run
tests/         gradient check and unit tests
data/          Dry_Bean_Dataset.xlsx (tracked in git; DryBeanDataset/ holds the original .arff/.txt/.xlsx)
```

The core depends only on numpy, pandas, scikit-learn and openpyxl. Plotting and web code live outside it.

## Setup

```bash
pip install -r requirements.txt   # or: pip install -e ".[dashboard,dev]"
```

Run everything from the repository root. The CLI's default `--data` path (`data/Dry_Bean_Dataset.xlsx`) is relative; the dashboard and examples locate the data file themselves.

## Usage

```bash
python -m drybean.cli --epochs 100          # train and evaluate on Dry Bean
python -m drybean.cli --hidden 64 49 --lr 0.01 --save-weights weights.npz
python -m dashboard.app                     # live dashboard at http://127.0.0.1:5000
python -m pytest                            # gradient check and unit tests (8 tests)
```

With `pip install -e .` the training command is also available as `drybean-train`.

Examples:

```bash
python -m examples.single_neuron_gd     # step-by-step forward/backward on a tiny network
python -m examples.experiments          # architecture and optimizer comparisons
python -m examples.train_with_plots     # spiral + Dry Bean runs with saved figures (written to figures/, which is gitignored)
```

## Data and evaluation

Dry Bean is split with stratification into 70% train, 15% validation and 15% test, and the scaler is fit on the training split only. The dashboard shows live train and validation loss and accuracy. The test set is scored once, after training finishes, and that final result is shown on its own cards.

Training uses early stopping on validation loss: it stops after 20 epochs without improvement and restores the best weights (`--patience 0` turns it off in the CLI). The dashboard stops the same way and says so when it finishes. `train_mini_batch` has early stopping off unless `patience` is passed.

The spiral (`examples/spiral.py`) is a synthetic 3-class problem used as a sanity check that the network can learn non-linear boundaries. It is not part of the classifier. The dashboard's spiral tab trains on 1000 points per class, validates on a separate fresh set of 100 per class, and scores a third fresh set of 100 per class once at the end (the generator adds new noise on every call). With only 100 points per class the network memorizes the noise.

The dashboard's Dry Bean tab trains for at most 50 epochs; the CLI defaults to 100.

`examples/train_with_plots.py` and `examples/experiments.py` predate this setup: they use 100 points per class, validate on the training data itself, and have no early stopping. Their spiral accuracies are training-set numbers, not held-out ones, and the three boundary plots in `train_with_plots.py` are all drawn after training, so they show the same final model.

Typical results, single runs with seed 42 and Adam:

| Dataset | Setup | Stopped at (best epoch) | Test acc |
|---|---|---|---|
| Dry Bean | `python -m drybean.cli` (hidden 64 32, lr 0.001, patience 20) | 46 (26) | ~92% (train ~94%, val ~93%) |
| Spiral | dashboard setup (hidden 64, lr 0.01, 1000 per class, patience 20) | 183 (163) | ~83% |

The spiral arms overlap near the centre because of the noise, so its test accuracy is capped well below 100%. Results vary with the seed and with how many points the test set has (300 for the spiral, 2042 for Dry Bean).

## Design notes

- He initialization, weights shaped `(n_inputs, n_neurons)`, batch dimension first.
- Softmax and cross-entropy are combined, so the logit gradient is `(y_pred - y_one_hot) / batch_size`.
- Mini-batches are reshuffled every epoch.
- Plain loops and NumPy on purpose, so each step is easy to follow.
