*Neural Network From Scratch – Iris Classification
** Project Overview

This project implements a fully connected feedforward neural network from scratch using only NumPy.
No deep learning frameworks (TensorFlow, PyTorch, Keras) were used.
The model is trained on the Iris dataset to classify flowers into three species.
The goal of this project is to deeply understand:
  *Forward propagation
  *Backpropagation (chain rule)
  *Gradient descent
  *Matrix-based gradient computation

***Model Architecture

Network structure:

Input Layer (4 features)
        ↓
Hidden Layer (4 neurons, Sigmoid)
        ↓
Output Layer (3 neurons, Sigmoid)

Forward Propagation
Z1 = W1 · X + b1
A1 = sigmoid(Z1)

Z2 = W2 · A1 + b2
Output = sigmoid(Z2)


Matrix dimensions:

X      → (4, m)
W1     → (4, 4)
b1     → (4, 1)
W2     → (3, 4)
b2     → (3, 1)
Output → (3, m)


m = number of training samples

***Backpropagation

Update rule:
W = W - learning_rate * gradient

Loss function (Mean Squared Error):
L = mean((Y - Y_hat)^2)

Derivative of loss:
dL/dY_hat = (2/m) * (Y_hat - Y)

Output Layer
delta2 = dL/dY_hat * sigmoid_derivative(Z2)

dW2 = delta2 · A1^T
db2 = sum(delta2)

Hidden Layer
delta1 = (W2^T · delta2) * sigmoid_derivative(Z1)

dW1 = delta1 · X^T
db1 = sum(delta1)

All gradient shapes match their parameter shapes exactly.

***Dataset

Iris Dataset
150 samples
4 numerical features
3 classes (one-hot encoded)

Train/Test split:
80% training
20% testing

***Training Configuration

Epochs: 2000
Learning rate: 0.1
Activation: Sigmoid
Loss: Mean Squared Error
Weight initialization: Random normal
Random seed: 42

***Evaluation

Accuracy is calculated using:
accuracy_score(true_labels, predictions)
Train and test accuracy are printed every 200 epochs.

***Decision Boundary Visualization

To visualize classification:
Two features are selected
Remaining features are set to zero
A mesh grid is generated
Predictions are made over the grid
Contour plot shows decision regions
This helps analyze how the learned weights separate classes.

*How to Run
pip install numpy pandas matplotlib seaborn scikit-learn
python creating_nn.py
Make sure Iris.csv is in the same directory.

*What This Project Demonstrates

Manual implementation of forward propagation
Full backpropagation using chain rule
Matrix-based gradient updates
Multi-class classification without frameworks
Understanding of gradient flow

***Future Improvements

Replace MSE with Cross-Entropy
Use Softmax in output layer
Add ReLU activation
Implement Mini-batch training
Add Xavier/He initialization
Compare with PyTorch implementation
