# Numpy MLP
This repository is designed to meet the requirements of the homework for 18-786 Introduction to Deep Learning HW2. It focuses on the implementation of backpropagation (BP) and the training of multilayer perceptrons (MLPs) using various optimizers.
## Implementation Features
The implementation offers the following functionalities:
- [x] Ability to handle any number of layers with different number of perceptrons per layer.
- [x] The following activation functions: ReLU, Sigmoid, tanh, and linear
- [x] Two potential loss functions for training: L2 and cross-entropy (CE)
- [x] Initialization of MLP parameters using either Xavier or He techniques
- [x] Execution of the forward pass in the MLP
- [x] Execution of the backward pass (also known as BP)
- [x] Employment of basic gradient descent for network training, with options for constant or decaying step size
- [x] Application of gradient descent with momentum for network training
- [x] Use of ADAM optimizer for network training