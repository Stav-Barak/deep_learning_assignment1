# deep_learning_assignment1

This repository contains the implementation of Assignment 1 in the Deep Learning course at Ben-Gurion University.  
The project implements a fully connected neural network from scratch, using PyTorch, to classify handwritten digits from the MNIST dataset.  

The assignment focuses on gaining a deep understanding of forward and backward propagation, while experimenting with:
- Batch Normalization
- L2 Regularization
- Comparison between models with and without these techniques

---

## Project Structure

- `mnist_classification.ipynb` – Jupyter Notebook with the full implementation and experiments  
- `mnist_classification_report.pdf` – Report summarizing design, implementation details, and results  

---

## Implementation Details

- Network Architecture:  
  - Input: 784 features (flattened MNIST images)  
  - Hidden layers: 20 → 7 → 5 neurons (ReLU activation)  
  - Output: 10 neurons (Softmax activation for classification)  

- Initialization:  
  - Weights initialized from a Gaussian distribution  
  - Biases initialized to zero  

- Training Settings:  
  - Learning rate: `0.009`  
  - Early stopping: Training stops if validation cost does not improve for 100 steps  
  - Batch size: chosen to balance speed and memory usage  

- Techniques Implemented:  
  - Forward propagation (`linear_forward`, `relu`, `softmax`, etc.)  
  - Backward propagation (`linear_backward`, `relu_backward`, `softmax_backward`)  
  - Cost computation with cross-entropy loss  
  - L2 regularization  
  - Batch normalization  

---

## Setup

Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib
```

---

## Run
1. Open the Jupyter notebook: jupyter notebook mnist_classification.ipynb.
2. Run all cells to train the model.
3. The notebook will output training progress, validation results, and final test accuracy.

---
