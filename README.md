# Neural Network Training — Spiral Data Classification

A hands-on example of training a simple feedforward neural network with **TensorFlow/Keras** to classify synthetic 2D spiral data into multiple classes. This project demonstrates the full ML workflow: data generation, preprocessing, model building, training, evaluation, and visualization.

## 📌 Overview

Spiral datasets are a classic toy problem in machine learning because they are **not linearly separable** — a simple linear classifier cannot solve them, making them a good test for neural networks. This notebook:

1. Generates a synthetic 3-class spiral dataset from scratch (no external dataset needed).
2. Splits the data into training and test sets.
3. Scales the features using `StandardScaler`.
4. Builds a small fully-connected neural network with Keras.
5. Trains the model and plots loss/accuracy curves.
6. Visualizes true vs. predicted classes on the test set.
7. Demonstrates inference on a new, unseen data point.

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `neural-network-training.ipynb` | Jupyter notebook containing the full workflow — data generation, model training, and visualizations. |

## 🧠 Model Architecture

A simple `Sequential` Keras model:

```
Input (2 features)
   → Dense(64, activation='relu')
   → Dense(32, activation='relu')
   → Dense(3, activation='softmax')   # 3 output classes
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
- **Epochs:** 50

## 📊 Dataset

The dataset is generated programmatically — a 3-armed spiral with 333 points per class (999 points total), each perturbed with Gaussian noise to simulate a realistic, non-trivial classification boundary.

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### Run the notebook

```bash
git clone https://github.com/pashajamal/Neural-Network-Training.git
cd Neural-Network-Training
jupyter notebook neural-network-training.ipynb
```

Run all cells in order to:
- Generate and visualize the spiral dataset
- Train the neural network
- View training/validation loss and accuracy curves
- Compare true vs. predicted class scatter plots
- Test the model on a custom new data point

## 📈 What You'll See

- A scatter plot of the raw spiral dataset colored by class.
- Training curves showing loss and accuracy over 50 epochs (train vs. validation).
- Side-by-side scatter plots comparing ground-truth labels to model predictions.
- A final plot showing how the trained model classifies a brand-new, unseen point.

## 🛠️ Tech Stack

- Python
- NumPy
- Matplotlib
- scikit-learn (train/test split, feature scaling)
- TensorFlow / Keras (model building and training)

## 📄 License

No license specified yet — consider adding one (e.g., MIT) if you plan to share or accept contributions.

## 🤝 Contributing

This is currently a personal/educational project. Feel free to fork it and experiment with different architectures, datasets, or hyperparameters.
