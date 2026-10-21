from flask import Flask, render_template_string, jsonify, request
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from neural_network import NeuralNetwork
from losses import Loss_CategoricalCrossentropy
from optimizer import Optimizer_Adam
from data import create_spiral_data, load_dry_bean

app = Flask(__name__)

training_state = {
    'spiral': {'history': None, 'is_training': False, 'model': None, 'X': None, 'y': None},
    'bean': {'history': None, 'is_training': False, 'model': None, 'X_val': None, 'y_val': None}
}

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return img_base64

def plot_architecture(layer_sizes):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    num_layers = len(layer_sizes)
    layer_positions = np.linspace(0, num_layers - 1, num_layers)

    i = 0
    while i < num_layers:
        layer_size = layer_sizes[i]
        x_pos = layer_positions[i]

        if layer_size <= 10:
            neuron_positions = np.linspace(0, 1, layer_size)
            j = 0
            while j < layer_size:
                y_pos = neuron_positions[j]
                circle = plt.Circle((x_pos, y_pos), 0.05, color='#ffffff', alpha=0.8, edgecolor='#d4af37', linewidth=2)
                ax.add_patch(circle)
                j = j + 1
        else:
            neuron_positions = np.linspace(0, 1, min(5, layer_size))
            j = 0
            while j < len(neuron_positions):
                y_pos = neuron_positions[j]
                circle = plt.Circle((x_pos, y_pos), 0.05, color='#ffffff', alpha=0.8, edgecolor='#d4af37', linewidth=2)
                ax.add_patch(circle)
                j = j + 1
            ax.text(x_pos + 0.1, 0.5, f"... ({layer_size})", fontsize=10, va='center', color='#d4af37')

        i = i + 1

    ax.set_xlim(-0.5, num_layers - 0.5 + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Architecture: {layer_sizes}', fontsize=14, fontweight='bold', color='#d4af37', pad=20)

    return fig_to_base64(fig)

def plot_weights_heatmap(model):
    weight_images = []
    layer_idx = 0

    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if hasattr(layer, 'weights'):
            weights = layer.weights

            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')

            im = ax.imshow(weights.T, cmap='gray', aspect='auto')
            ax.set_xlabel('Input Features', color='#d4af37', fontsize=11)
            ax.set_ylabel('Neurons', color='#d4af37', fontsize=11)
            ax.set_title(f'Layer {layer_idx} Weight Heatmap ({weights.shape[0]} inputs → {weights.shape[1]} neurons)',
                        fontsize=14, fontweight='bold', color='#d4af37', pad=20)
            ax.tick_params(colors='#d4af37')

            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(colors='#d4af37')

            weight_images.append(fig_to_base64(fig))
            layer_idx = layer_idx + 1

        i = i + 1

    return weight_images if weight_images else [None]

def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min = X[:, 0].min() - 0.1
    x_max = X[:, 0].max() + 0.1
    y_min = X[:, 1].min() - 0.1
    y_max = X[:, 1].max() + 0.1

    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)
    XX, YY = np.meshgrid(xx, yy)

    Z_input = np.c_[XX.ravel(), YY.ravel()]
    Z_pred = model.forward(Z_input)
    Z = np.argmax(Z_pred, axis=1)
    Z = Z.reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    ax.contourf(XX, YY, Z, alpha=0.4, cmap='gray', levels=3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='gray', edgecolors='#d4af37', s=100, alpha=0.8, linewidth=2)

    ax.set_xlabel('Feature 1', color='#d4af37', fontsize=11)
    ax.set_ylabel('Feature 2', color='#d4af37', fontsize=11)
    ax.set_title('Decision Boundary', fontsize=14, fontweight='bold', color='#d4af37', pad=20)
    ax.tick_params(colors='#d4af37')

    return fig_to_base64(fig)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background: #0a0a0a;
            min-height: 100vh;
            padding: 20px;
            color: #ffffff;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: #1a1a1a;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            overflow: hidden;
            border: 1px solid #333333;
        }

        .header {
            background: #1a1a1a;
            color: #d4af37;
            padding: 50px 40px;
            text-align: center;
            border-bottom: 3px solid #d4af37;
        }

        .header h1 {
            font-size: 42px;
            margin-bottom: 10px;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .header p {
            font-size: 16px;
            opacity: 0.85;
            font-weight: 400;
            color: #cccccc;
        }

        .tabs {
            display: flex;
            background: #0a0a0a;
            border-bottom: 1px solid #333333;
            overflow-x: auto;
        }

        .tab-button {
            flex: 1;
            padding: 18px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            color: #888888;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
            white-space: nowrap;
        }

        .tab-button:hover {
            color: #d4af37;
            background: rgba(212, 175, 55, 0.05);
        }

        .tab-button.active {
            color: #d4af37;
            border-bottom: 3px solid #d4af37;
            background: rgba(212, 175, 55, 0.1);
        }

        .content {
            padding: 40px;
            min-height: 600px;
            background: #1a1a1a;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .tab-content.active {
            display: block;
        }

        .tab-content h2 {
            font-size: 28px;
            margin-bottom: 30px;
            color: #d4af37;
            font-weight: 700;
        }

        .section {
            margin-bottom: 35px;
        }

        .section-title {
            color: #d4af37;
            font-size: 16px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 20px;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 25px;
            background: #2a2a2a;
            border-radius: 10px;
            border: 1px solid #333333;
        }

        .control-group {
            display: flex;
            flex-direction: column;
        }

        .control-group label {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 13px;
            color: #d4af37;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .control-group input,
        .control-group select {
            padding: 12px 15px;
            border: 2px solid #333333;
            border-radius: 8px;
            font-size: 14px;
            background: #1a1a1a;
            color: #ffffff;
            transition: all 0.3s;
        }

        .control-group input:focus,
        .control-group select:focus {
            outline: none;
            border-color: #d4af37;
            box-shadow: 0 0 10px rgba(212, 175, 55, 0.2);
            background: #2a2a2a;
        }

        .layers-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #2a2a2a;
            border-radius: 10px;
            border: 1px solid #333333;
            margin-bottom: 25px;
        }

        .button {
            padding: 14px 32px;
            background: #8b7d3f;
            color: #000000;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 700;
            font-size: 15px;
            transition: color 0.3s;
        }

        .button:hover:not(:disabled) {
            color: #ffffff;
        }

        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .tab-content h2,
        .tab-content h3 {
            margin-top: 50px;
            margin-bottom: 25px;
        }

        .tab-content ul,
        .tab-content ol {
            margin-bottom: 50px;
        }

        .tab-content li {
            margin-bottom: 15px;
            line-height: 1.6;
        }

        .plot-description {
            margin-bottom: 30px;
        }

        .plot {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #333333;
        }

        .plot img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        .plot-description {
            background: #2a2a2a;
            border-left: 4px solid #d4af37;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 14px;
            line-height: 1.6;
            color: #cccccc;
            border: 1px solid #333333;
        }

        .live-metrics {
            display: none;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }

        .live-metrics.active {
            display: flex;
        }

        .metric-card {
            flex: 1;
            min-width: 140px;
            background: #1a1a1a;
            border: 1px solid #d4af37;
            border-radius: 6px;
            padding: 12px 16px;
            text-align: center;
        }

        .metric-label {
            font-size: 11px;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #888888;
        }

        .metric-value {
            font-size: 26px;
            font-weight: bold;
            color: #d4af37;
            font-variant-numeric: tabular-nums;
            margin-top: 4px;
        }

        .status {
            padding: 18px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 600;
            border-left: 4px solid;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .status.training {
            background: rgba(212, 175, 55, 0.1);
            color: #d4af37;
            border-color: #d4af37;
        }

        .status.complete {
            background: rgba(212, 175, 55, 0.1);
            color: #d4af37;
            border-color: #d4af37;
        }

        .info-box {
            background: #2a2a2a;
            border-left: 4px solid #d4af37;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #333333;
        }

        .info-box strong {
            color: #d4af37;
        }

        .info-box ul {
            margin-left: 20px;
            margin-top: 10px;
        }

        .info-box li {
            margin: 8px 0;
            color: #cccccc;
        }

        .info-box {
            margin-bottom: 40px;
        }

        .tab-content p {
            margin-bottom: 25px;
        }

        h3 {
            color: #d4af37;
            margin-top: 25px;
            margin-bottom: 15px;
        }

        p {
            color: #cccccc;
            line-height: 1.6;
            margin-bottom: 15px;
        }

        code {
            background: rgba(212, 175, 55, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            color: #d4af37;
            font-family: 'Courier New', monospace;
        }

        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(212, 175, 55, 0.05);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(212, 175, 55, 0.3);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(212, 175, 55, 0.5);
        }

        input[type="range"] {
            accent-color: #8b7d3f;
        }

        @supports (accent-color: #8b7d3f) {
            input[type="range"] {
                accent-color: #8b7d3f;
            }
        }

        input[type="range"]::-webkit-slider-thumb {
            accent-color: #8b7d3f;
            background: #8b7d3f;
        }

        input[type="range"]::-moz-range-thumb {
            background: #8b7d3f;
            border-color: #8b7d3f;
        }

        input[type="range"]::-moz-range-track {
            background: #333333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Neural Network Lab</h1>
            <p>Interactive Deep Learning Visualization Platform</p>
        </div>

        <div class="tabs">
            <button class="tab-button active" onclick="switchTab(0)">Spiral Training</button>
            <button class="tab-button" onclick="switchTab(1)">Bean Training</button>
            <button class="tab-button" onclick="switchTab(2)">Visualizations</button>
            <button class="tab-button" onclick="switchTab(3)">About</button>
        </div>

        <div class="content">
            <!-- TAB 0: Spiral Training -->
            <div class="tab-content active">
                <h2>Spiral Dataset</h2>

                <div class="section">
                    <div class="section-title">Configuration</div>
                    <div class="controls">
                        <div class="control-group">
                            <label>Hidden Layers</label>
                            <input type="number" id="spiral_num_layers" min="1" max="5" value="1" onchange="updateSpiralLayers()">
                        </div>
                        <div class="control-group">
                            <label>Learning Rate</label>
                            <select id="spiral_lr">
                                <option value="0.001">0.001</option>
                                <option value="0.01" selected>0.01</option>
                                <option value="0.1">0.1</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label>Epochs</label>
                            <input type="range" id="spiral_epochs" min="100" max="1000" step="100" value="500">
                            <span style="font-size: 12px; color: #d4af37; margin-top: 5px;" id="spiral_epochs_val">500</span>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Layer Sizes</div>
                    <div id="spiral_layers_inputs" class="layers-container"></div>
                </div>

                <button class="button" id="spiral_train_btn" onclick="trainSpiral()">Train Network</button>

                <div id="spiral_status" class="status" style="display:none;"></div>
                <div id="spiral_metrics" class="live-metrics"></div>
                <div id="spiral_loss" class="plot"></div>
                <div id="spiral_acc" class="plot"></div>
                <div id="spiral_arch" class="plot"></div>
                <div id="spiral_boundary" class="plot"></div>
                <div id="spiral_weights" class="plot"></div>
            </div>

            <!-- TAB 1: Dry Bean Training -->
            <div class="tab-content">
                <h2>Dry Bean Dataset</h2>

                <div class="section">
                    <div class="section-title">Configuration</div>
                    <div class="controls">
                        <div class="control-group">
                            <label>Hidden Layers</label>
                            <input type="number" id="bean_num_layers" min="1" max="5" value="2" onchange="updateBeanLayers()">
                        </div>
                        <div class="control-group">
                            <label>Learning Rate</label>
                            <select id="bean_lr">
                                <option value="0.0001">0.0001</option>
                                <option value="0.001" selected>0.001</option>
                                <option value="0.01">0.01</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Layer Sizes</div>
                    <div id="bean_layers_inputs" class="layers-container"></div>
                </div>

                <button class="button" id="bean_train_btn" onclick="trainBean()">Train Network</button>

                <div id="bean_status" class="status" style="display:none;"></div>
                <div id="bean_metrics" class="live-metrics"></div>
                <div id="bean_loss" class="plot"></div>
                <div id="bean_acc" class="plot"></div>
                <div id="bean_arch" class="plot"></div>
                <div id="bean_weights" class="plot"></div>
            </div>

            <!-- TAB 2: Visualizations -->
            <div class="tab-content">
                <h2>Visualizations & Results</h2>

                <h3>Architecture Diagram</h3>
                <div class="plot-description">
                    <strong>What it shows:</strong> The network structure with neurons (white circles) and connections. Each layer represents a stage of computation. Larger numbers indicate more neurons in that layer, allowing the network to learn more complex patterns.
                </div>

                <h3>Training Loss Curve</h3>
                <div class="plot-description">
                    <strong>What it shows:</strong> How the prediction error decreases over epochs. A steadily declining loss indicates the network is learning. Flat curves suggest learning has plateaued or the learning rate needs adjustment.
                </div>

                <h3>Training Accuracy Curve</h3>
                <div class="plot-description">
                    <strong>What it shows:</strong> The percentage of correct predictions over training. Accuracy rising together with loss decreasing confirms the network is improving its classification performance.
                </div>

                <h3>Decision Boundary (Spiral Only)</h3>
                <div class="plot-description">
                    <strong>What it shows:</strong> How the trained network divides the 2D input space into regions for each class. Black regions are decision boundaries. Points marked are training samples colored by their true class. The network "learns" by adjusting these boundaries.
                </div>

                <h3>Weight Heatmap</h3>
                <div class="plot-description">
                    <strong>What it shows:</strong> The first layer's weights connecting inputs to hidden neurons. Light areas = positive weights (strengthening that connection), dark areas = negative weights (weakening that connection). This shows what patterns the network extracts from raw inputs.
                </div>

                <h3>How to Interpret Combined Results</h3>
                <div class="info-box">
                    <strong>Good training:</strong> Loss continuously decreases, accuracy increases, decision boundary becomes smoother and more confident, weight patterns become more distinct.
                    <br><br>
                    <strong>Bad training:</strong> Loss stays flat or increases, accuracy plateaus or decreases, decision boundary remains chaotic, weights show little change.
                    <br><br>
                    <strong>Overfitting:</strong> Training metrics improve but validation metrics worsen, decision boundary becomes overly complex with many small regions.
                </div>
            </div>

            <!-- TAB 3: About -->
            <div class="tab-content">
                <h2>About</h2>

                <h3>What is This?</h3>
                <p>
                    A complete neural network implementation built from scratch with NumPy. Features configurable architectures, verified backpropagation, and interactive real-time visualization.
                </p>

                <h3>Features</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Dynamic layer configuration (1-5 hidden layers with custom sizes)</li>
                    <li>Real-time training visualization with loss and accuracy curves</li>
                    <li>Architecture diagrams automatically generated from your configuration</li>
                    <li>Weight heatmaps showing what the network learns</li>
                    <li>Decision boundary visualization (Spiral dataset)</li>
                    <li>Multiple optimizers: SGD, Momentum, and Adam</li>
                    <li>Verified backpropagation with gradient checks (error < 1e-9)</li>
                </ul>

                <h3>Datasets</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Spiral:</strong> 300 samples, 3 classes, 2D features - Perfect for visualizing how networks learn non-linear boundaries</li>
                    <li><strong>Dry Bean:</strong> 13,611 samples, 7 varieties, 16 features - Real-world classification task</li>
                </ul>

                <h3>How It Works</h3>
                <p>
                    1. Configure your network architecture (number and size of hidden layers)
                    2. Choose learning rate and number of epochs
                    3. Click "Train Network" to start training
                    4. Watch the loss decrease and accuracy increase in real-time
                    5. Inspect the learned weights and decision boundaries in the Visualizations tab
                </p>
            </div>
        </div>
    </div>

    <script>
        Plotly.setPlotConfig({responsive: true});

        function switchTab(index) {
            const tabs = document.querySelectorAll('.tab-content');
            const buttons = document.querySelectorAll('.tab-button');
            tabs.forEach((tab, i) => {
                tab.classList.toggle('active', i === index);
            });
            buttons.forEach((btn, i) => {
                btn.classList.toggle('active', i === index);
            });
        }

        function updateSpiralLayers() {
            const numLayers = parseInt(document.getElementById('spiral_num_layers').value);
            const container = document.getElementById('spiral_layers_inputs');
            container.innerHTML = '';

            let i = 0;
            while (i < numLayers) {
                const div = document.createElement('div');
                div.className = 'control-group';
                div.innerHTML = `<label>Layer ${i+1}</label><input type="number" id="spiral_hidden_${i}" min="8" max="256" value="${64 - i*10}" />`;
                container.appendChild(div);
                i = i + 1;
            }
        }

        function updateBeanLayers() {
            const numLayers = parseInt(document.getElementById('bean_num_layers').value);
            const container = document.getElementById('bean_layers_inputs');
            container.innerHTML = '';

            let i = 0;
            while (i < numLayers) {
                const div = document.createElement('div');
                div.className = 'control-group';
                div.innerHTML = `<label>Layer ${i+1}</label><input type="number" id="bean_hidden_${i}" min="8" max="256" value="${64 - i*15}" />`;
                container.appendChild(div);
                i = i + 1;
            }
        }

        document.getElementById('spiral_epochs').addEventListener('input', (e) => {
            document.getElementById('spiral_epochs_val').textContent = e.target.value;
        });

        updateSpiralLayers();
        updateBeanLayers();

        function renderLiveMetrics(prefix, m) {
            const box = document.getElementById(prefix + '_metrics');
            if (!box || !m) return;
            const fmt = (v, d) => (v === null || v === undefined) ? '-' : v.toFixed(d);
            const cards = [['Epoch', m.epoch.toString()], ['Loss', fmt(m.loss, 4)], ['Accuracy', fmt(m.acc * 100, 2) + '%']];
            if (m.val_loss !== undefined) {
                cards.push(['Val Loss', fmt(m.val_loss, 4)], ['Val Accuracy', fmt(m.val_acc * 100, 2) + '%']);
            }
            cards.push(['Test Loss', fmt(m.test_loss, 4)], ['Test Accuracy', fmt(m.test_acc * 100, 2) + '%']);
            box.innerHTML = cards.map(c => `<div class="metric-card"><div class="metric-label">${c[0]}</div><div class="metric-value">${c[1]}</div></div>`).join('');
            box.classList.add('active');
        }

        async function updateSpiralCharts() {
            const response = await fetch('/training_progress/spiral');
            const data = await response.json();
            const status = document.getElementById('spiral_status');

            if (data.loss_plot && data.loss_plot !== null) {
                const loss_fig = JSON.parse(data.loss_plot);
                Plotly.newPlot('spiral_loss', loss_fig.data, loss_fig.layout);
            }

            if (data.acc_plot && data.acc_plot !== null) {
                const acc_fig = JSON.parse(data.acc_plot);
                Plotly.newPlot('spiral_acc', acc_fig.data, acc_fig.layout);
            }

            renderLiveMetrics('spiral', data.metrics);

            if (data.is_training && status && data.current_epoch) {
                status.textContent = `Training... Epoch ${data.current_epoch}/${document.getElementById('spiral_epochs').value}`;
            }

            return data.is_training;
        }

        async function trainSpiral() {
            const numLayers = parseInt(document.getElementById('spiral_num_layers').value);
            const lr = document.getElementById('spiral_lr').value;
            const epochs = document.getElementById('spiral_epochs').value;

            const layerSizes = [2];
            let i = 0;
            while (i < numLayers) {
                layerSizes.push(parseInt(document.getElementById(`spiral_hidden_${i}`).value));
                i = i + 1;
            }
            layerSizes.push(3);

            const btn = document.getElementById('spiral_train_btn');
            btn.disabled = true;

            const status = document.getElementById('spiral_status');
            status.style.display = 'block';
            status.classList.add('training');
            status.textContent = 'Training... Architecture: ' + JSON.stringify(layerSizes);

            const trainRequest = fetch('/train_spiral', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({layerSizes, lr, epochs})
            }).then(response => response.json()).then(data => {
                if (data.arch_img) {
                    document.getElementById('spiral_arch').innerHTML = `<h3>Network Architecture</h3><img src="data:image/png;base64,${data.arch_img}" />`;
                }

                if (data.boundary_img) {
                    document.getElementById('spiral_boundary').innerHTML = `<h3>Decision Boundary</h3><img src="data:image/png;base64,${data.boundary_img}" />`;
                }

                if (data.weights_img && data.weights_img.length > 0) {
                    let weightsHtml = '<h3>Weight Heatmaps</h3>';
                    let idx = 0;
                    while (idx < data.weights_img.length) {
                        if (data.weights_img[idx]) {
                            weightsHtml += `<div style="margin-bottom: 30px;"><img src="data:image/png;base64,${data.weights_img[idx]}" style="width: 100%; max-width: 800px;" /></div>`;
                        }
                        idx = idx + 1;
                    }
                    document.getElementById('spiral_weights').innerHTML = weightsHtml;
                }

                status.classList.remove('training');
                status.classList.add('complete');
                status.textContent = `Training complete! Test Accuracy: ${(data.test_acc * 100).toFixed(2)}%`;

                btn.disabled = false;
            });

            let done = false;
            trainRequest.finally(() => { done = true; });
            while (!done) {
                await new Promise(resolve => setTimeout(resolve, 500));
                await updateSpiralCharts();
            }
            await updateSpiralCharts();
        }

        async function updateBeanCharts() {
            const response = await fetch('/training_progress/bean');
            const data = await response.json();
            const status = document.getElementById('bean_status');

            if (data.loss_plot && data.loss_plot !== null) {
                const loss_fig = JSON.parse(data.loss_plot);
                Plotly.newPlot('bean_loss', loss_fig.data, loss_fig.layout);
            }

            if (data.acc_plot && data.acc_plot !== null) {
                const acc_fig = JSON.parse(data.acc_plot);
                Plotly.newPlot('bean_acc', acc_fig.data, acc_fig.layout);
            }

            renderLiveMetrics('bean', data.metrics);

            if (data.is_training && status && data.current_epoch) {
                status.textContent = `Training... Epoch ${data.current_epoch}/50`;
            }

            return data.is_training;
        }

        async function trainBean() {
            const numLayers = parseInt(document.getElementById('bean_num_layers').value);
            const lr = document.getElementById('bean_lr').value;

            const layerSizes = [16];
            let i = 0;
            while (i < numLayers) {
                layerSizes.push(parseInt(document.getElementById(`bean_hidden_${i}`).value));
                i = i + 1;
            }
            layerSizes.push(7);

            const btn = document.getElementById('bean_train_btn');
            btn.disabled = true;

            const status = document.getElementById('bean_status');
            status.style.display = 'block';
            status.classList.add('training');
            status.textContent = 'Training... Architecture: ' + JSON.stringify(layerSizes);

            const trainRequest = fetch('/train_bean', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({layerSizes, lr, epochs: 50})
            }).then(response => response.json()).then(data => {
                if (data.arch_img) {
                    document.getElementById('bean_arch').innerHTML = `<h3>Network Architecture</h3><img src="data:image/png;base64,${data.arch_img}" />`;
                }

                if (data.weights_img && data.weights_img.length > 0) {
                    let weightsHtml = '<h3>Weight Heatmaps</h3>';
                    let idx = 0;
                    while (idx < data.weights_img.length) {
                        if (data.weights_img[idx]) {
                            weightsHtml += `<div style="margin-bottom: 30px;"><img src="data:image/png;base64,${data.weights_img[idx]}" style="width: 100%; max-width: 800px;" /></div>`;
                        }
                        idx = idx + 1;
                    }
                    document.getElementById('bean_weights').innerHTML = weightsHtml;
                }

                status.classList.remove('training');
                status.classList.add('complete');
                status.textContent = `Training complete! Test Accuracy: ${data.test_acc.toFixed(4)}`;

                btn.disabled = false;
            });

            let done = false;
            trainRequest.finally(() => { done = true; });
            while (!done) {
                await new Promise(resolve => setTimeout(resolve, 500));
                await updateBeanCharts();
            }
            await updateBeanCharts();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/training_progress/<dataset>', methods=['GET'])
def get_training_progress(dataset):
    if dataset not in training_state:
        return jsonify({'loss_plot': None, 'acc_plot': None, 'is_training': False, 'current_epoch': 0})

    state = training_state[dataset]
    if state['history'] is None or len(state['history'].get('loss', state['history'].get('train_loss', []))) == 0:
        return jsonify({'loss_plot': None, 'acc_plot': None, 'is_training': state['is_training'], 'current_epoch': 0})

    history = state['history']
    loss_plot = None
    acc_plot = None

    if dataset == 'spiral':
        if len(history['loss']) > 0:
            epochs_arr = list(range(len(history['loss'])))
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(x=epochs_arr, y=history['loss'], mode='lines', name='Train', line=dict(color='#d4af37', width=3)))
            loss_fig.add_trace(go.Scatter(x=epochs_arr, y=history['test_loss'], mode='lines', name='Test', line=dict(color='#888888', width=3)))
            loss_fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss', height=400, plot_bgcolor='#1a1a1a', paper_bgcolor='#2a2a2a', font=dict(color='#d4af37'))
            loss_plot = json.dumps(loss_fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

            acc_fig = go.Figure()
            acc_fig.add_trace(go.Scatter(x=epochs_arr, y=history['acc'], mode='lines', name='Train', line=dict(color='#d4af37', width=3)))
            acc_fig.add_trace(go.Scatter(x=epochs_arr, y=history['test_acc'], mode='lines', name='Test', line=dict(color='#888888', width=3)))
            acc_fig.update_layout(title='Training Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', height=400, plot_bgcolor='#1a1a1a', paper_bgcolor='#2a2a2a', font=dict(color='#d4af37'))
            acc_plot = json.dumps(acc_fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    else:
        if len(history['train_loss']) > 0:
            epochs_arr = list(range(len(history['train_loss'])))
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(x=epochs_arr, y=history['train_loss'], mode='lines', name='Train', line=dict(color='#d4af37', width=3)))
            loss_fig.add_trace(go.Scatter(x=epochs_arr, y=history['val_loss'], mode='lines', name='Val', line=dict(color='#888888', width=3)))
            loss_fig.add_trace(go.Scatter(x=epochs_arr, y=history['test_loss'], mode='lines', name='Test', line=dict(color='#4fc3f7', width=3)))
            loss_fig.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss', height=400, plot_bgcolor='#1a1a1a', paper_bgcolor='#2a2a2a', font=dict(color='#d4af37'))
            loss_plot = json.dumps(loss_fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

            acc_fig = go.Figure()
            acc_fig.add_trace(go.Scatter(x=epochs_arr, y=history['train_acc'], mode='lines', name='Train', line=dict(color='#d4af37', width=3)))
            acc_fig.add_trace(go.Scatter(x=epochs_arr, y=history['val_acc'], mode='lines', name='Val', line=dict(color='#888888', width=3)))
            acc_fig.add_trace(go.Scatter(x=epochs_arr, y=history['test_acc'], mode='lines', name='Test', line=dict(color='#4fc3f7', width=3)))
            acc_fig.update_layout(title='Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', height=400, plot_bgcolor='#1a1a1a', paper_bgcolor='#2a2a2a', font=dict(color='#d4af37'))
            acc_plot = json.dumps(acc_fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    current_epoch = len(history.get('loss', history.get('train_loss', [])))
    if dataset == 'spiral':
        metrics = {'epoch': current_epoch, 'loss': float(history['loss'][-1]), 'acc': float(history['acc'][-1]),
                   'test_loss': float(history['test_loss'][-1]), 'test_acc': float(history['test_acc'][-1])}
    else:
        metrics = {'epoch': current_epoch, 'loss': float(history['train_loss'][-1]), 'acc': float(history['train_acc'][-1]),
                   'val_loss': float(history['val_loss'][-1]), 'val_acc': float(history['val_acc'][-1]),
                   'test_loss': float(history['test_loss'][-1]), 'test_acc': float(history['test_acc'][-1])}
    return jsonify({
        'metrics': metrics,
        'loss_plot': loss_plot,
        'acc_plot': acc_plot,
        'is_training': state['is_training'],
        'current_epoch': current_epoch
    })

@app.route('/train_spiral', methods=['POST'])
def train_spiral():
    global training_state

    params = request.json
    layer_sizes = params['layerSizes']
    lr = float(params['lr'])
    epochs = int(params['epochs'])

    np.random.seed(42)
    X, y = create_spiral_data(samples=100, classes=3)
    X_test, y_test = create_spiral_data(samples=100, classes=3)

    model = NeuralNetwork(layer_sizes)
    optimizer = Optimizer_Adam(learning_rate=lr)
    loss_fn = Loss_CategoricalCrossentropy()

    history = {'loss': [], 'acc': [], 'test_loss': [], 'test_acc': []}

    training_state['spiral']['is_training'] = True
    training_state['spiral']['history'] = history
    training_state['spiral']['model'] = model
    training_state['spiral']['X'] = X
    training_state['spiral']['y'] = y

    epoch = 0
    while epoch < epochs:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_end = min(batch_start + 32, X.shape[0])
            batch_indices = indices[batch_start:batch_end]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            y_pred = model.forward(X_batch)
            model.backward(y_batch)
            optimizer.update_params(model.layers)

            batch_start = batch_end

        y_pred_full = model.forward(X)
        loss = loss_fn.calculate(y_pred_full, y)
        acc = np.mean(np.argmax(y_pred_full, axis=1) == y)

        y_pred_test = model.forward(X_test)
        test_loss = loss_fn.calculate(y_pred_test, y_test)
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == y_test)

        history['loss'].append(float(loss))
        history['acc'].append(float(acc))
        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))

        epoch = epoch + 1

    training_state['spiral']['is_training'] = False

    arch_img = plot_architecture(layer_sizes)
    boundary_img = plot_decision_boundary(model, X, y)
    weights_img = plot_weights_heatmap(model)

    return jsonify({
        'test_acc': history['test_acc'][-1],
        'arch_img': arch_img,
        'boundary_img': boundary_img,
        'weights_img': weights_img,
        'complete': True
    })

@app.route('/train_bean', methods=['POST'])
def train_bean():
    global training_state

    params = request.json
    layer_sizes = params['layerSizes']
    lr = float(params['lr'])

    np.random.seed(42)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
    training_state['bean']['is_training'] = True
    training_state['bean']['history'] = history

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dry_bean('data/Dry_Bean_Dataset.xlsx')
    except:
        training_state['bean']['is_training'] = False
        return jsonify({'error': 'Dataset not found'}), 400

    model = NeuralNetwork(layer_sizes)
    optimizer = Optimizer_Adam(learning_rate=lr)
    loss_fn = Loss_CategoricalCrossentropy()

    training_state['bean']['model'] = model
    training_state['bean']['X_val'] = X_val
    training_state['bean']['y_val'] = y_val

    epoch = 0
    while epoch < 50:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        batch_start = 0
        while batch_start < X_train.shape[0]:
            batch_end = min(batch_start + 64, X_train.shape[0])
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

        y_pred_test = model.forward(X_test)
        test_loss = loss_fn.calculate(y_pred_test, y_test)
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == y_test)

        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        epoch = epoch + 1

    training_state['bean']['is_training'] = False

    y_pred_test = model.forward(X_test)
    test_loss = loss_fn.calculate(y_pred_test, y_test)
    test_acc = np.mean(np.argmax(y_pred_test, axis=1) == y_test)

    arch_img = plot_architecture(layer_sizes)
    weights_img = plot_weights_heatmap(model)

    return jsonify({
        'arch_img': arch_img,
        'weights_img': weights_img,
        'test_acc': float(test_acc),
        'complete': True
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
