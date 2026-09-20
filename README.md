# ID3 Decision Tree + Neural Network from Scratch

Two machine-learning implementations built to explore core classification algorithms without relying on high-level ML frameworks.

The project includes:

- an **ID3 decision tree** for categorical car-evaluation data
- a **fully connected neural network** for Fashion-MNIST image classification
- custom training, evaluation and visualisation code using NumPy and Matplotlib

## Results

### ID3 decision tree

The decision tree is trained on the Car Evaluation dataset using entropy and information gain to select splits.

- 80/20 train-test split
- categorical feature handling
- recursive tree construction
- per-class precision, recall and F1 metrics
- learning-curve experiments

Observed test accuracy: **94.2%**.

### Neural network

The neural network is implemented from scratch with NumPy.

Architecture:

```text
784 inputs -> 30 hidden units -> 10 outputs
```

The implementation includes:

- Xavier/Glorot weight initialisation
- sigmoid activations
- forward propagation
- backpropagation
- mini-batch training
- hyperparameter experiments
- learning-curve visualisation

Best observed Fashion-MNIST test accuracy: **88.23%**.

## Repository structure

```text
decisionLearningTree/
  car.csv
  id3_decision_tree.py

neuralNetwork/
  nn.py
  fashion-mnist_train.csv.gz
  fashion-mnist_test.csv.gz
```

## Run the decision tree

```bash
cd decisionLearningTree
python id3_decision_tree.py
```

## Run the neural network

```bash
cd neuralNetwork
python nn.py 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
```

## Stack

- Python
- NumPy
- Matplotlib
- Decision trees
- Neural networks
- Classification metrics
