# Neural Network Lab - Minimum Working Version

A simple neural network built from scratch using NumPy for classifying the UCI Dry Bean dataset and visualizing the learning process.

## Files

### Core Components
- **neural_network_new.py** - Layer_Dense (forward/backward), NeuralNetwork, and activations (ReLU, Softmax, Sigmoid, Tanh)
- **losses_new.py** - Categorical Cross-Entropy loss
- **optimizer_new.py** - SGD (with optional momentum) and Adam optimizers
- **data_new.py** - Spiral data generator and Dry Bean dataset loader

### Training and Evaluation
- **train_new.py** - Mini-batch training loop with history tracking
- **gradient_check_new.py** - Numerical vs analytical gradient comparison

### Demos and Experiments
- **single_neuron_gd_new.py** - 3-layer, 1-neuron per layer demo with step-by-step math
- **experiments_new.py** - Architecture and optimizer comparisons
- **visualize_new.py** - Plotting functions for training curves, confusion matrix, weights, architecture

### Main Entry Points
- **main_new.py** - Full pipeline: trains on spiral, trains on Dry Bean, saves visualizations
- **test_spiral_new.py** - Quick test of forward pass on spiral data
- **test_training_new.py** - Quick test of training loop with backprop

## Quick Start

### Test Forward Pass
```bash
python test_spiral_new.py
```
Expected output: Loss ≈ 1.10, Accuracy ≈ 0.33

### Test Training Loop
```bash
python test_training_new.py
```
Expected: Loss decreases, accuracy increases

### Run Single Neuron Demo
```bash
python single_neuron_gd_new.py
```
Shows step-by-step forward and backward pass for a tiny 3-1-1-1 network

### Run Full Pipeline
```bash
python main_new.py
```
Trains on spiral and Dry Bean datasets, saves figures to `figures/` directory

### Run Experiments
```bash
python experiments_new.py
```
Compares different architectures and optimizers

## Design Decisions

**Weight Initialization:** He initialization - `randn(n_in, n_out) * sqrt(2 / n_in)`

**Weights Shape:** `(n_inputs, n_neurons)` - forward pass is `inputs @ weights + biases`

**Batch Dimension:** Always first - `(batch_size, features)`

**Combined Softmax + Cross-Entropy:** Gradient w.r.t logits is `(y_pred - y_one_hot) / batch_size`

**Backward Loop:** Reverses forward loop - works for any number of layers

**Loss:** Mean over batch, mini-batches reshuffled each epoch

**Preprocessing:** Stratified split → fit StandardScaler on train only → transform all

## Reference Results

### Spiral (100 samples × 3 classes)
- Untrained: loss ≈ 1.10, accuracy ≈ 0.33
- `[2, 64, 3]` + Adam (lr=0.01, 1000 epochs): ≈ 83% accuracy
- `[2, 64, 64, 64, 3]` + Adam (lr=0.01, 1000 epochs): ≈ 99% accuracy
- `[2, 64, 3]` + SGD (lr=1.0, 10000 epochs): ≈ 72% accuracy

## Code Style

- No list comprehensions - all loops are explicit
- Simple expressions - no fancy NumPy tricks
- Clear variable names
- Comments for key operations

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn
- openpyxl (for reading Excel files)

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## Structure

Files are named with `_new` suffix to distinguish from the original Parts 1-4 code. Copy to a new folder and rename without `_new` to use.
