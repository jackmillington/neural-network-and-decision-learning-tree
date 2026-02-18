# 2802ICT Intelligent Systems — Assignment 2 (Programming)

Author: **Jack Millington** (s5405915) :contentReference[oaicite:0]{index=0}

This repository contains my Assignment 2 programming work for **2802ICT Intelligent Systems**, covering:
- **Part 1:** ID3 Decision Tree classifier on `car.csv`
- **Part 2:** From-scratch 3-layer Neural Network on `fashion-mnist.csv.gz` :contentReference[oaicite:1]{index=1}

## Part 1 — ID3 Decision Tree (Car Evaluation)
- Loads `car.csv` (1728 rows, 6 categorical features, 4 classes), shuffles with a fixed seed, and splits **80/20** train/test. :contentReference[oaicite:2]{index=2}
- Implements entropy, information gain, partitioning, and a Node-based tree structure for the ID3 algorithm. :contentReference[oaicite:3]{index=3}
- Reports accuracy and per-class metrics (precision/recall/F1) and plots a learning curve. :contentReference[oaicite:4]{index=4}

**Result:** **94.2%** test accuracy (weighted F1 **0.943**, macro F1 **0.846**). :contentReference[oaicite:5]{index=5}

## Part 2 — Neural Network (Fashion-MNIST)
- Implements a **[784, 30, 10]** fully connected network with sigmoid activations.
- Trains using forward propagation + backpropagation, batching, Xavier/Glorot weight initialisation, and evaluates accuracy across experiments. :contentReference[oaicite:6]{index=6}

**Best result:** **88.23%** test accuracy using **learning rate = 1.0**, **batch size = 100**, **epochs = 100**. :contentReference[oaicite:7]{index=7}

## Report
The full write-up (design, experiments, results, discussion, and appendices with code) is in:
- `Jack_Millington_s5405915_A2_report.pdf` :contentReference[oaicite:8]{index=8}

## How to run (high level)
Exact filenames/commands depend on how you’ve arranged the repo, but the code in the report shows:
- Decision tree script expects `car.csv` in the working directory. :contentReference[oaicite:9]{index=9}
- Neural network is run as:
  `python nn.py NInput NHidden NOutput train.csv.gz test.csv.gz` :contentReference[oaicite:10]{index=10}

## Tech
- Python
- NumPy
- Matplotlib :contentReference[oaicite:11]{index=11}
