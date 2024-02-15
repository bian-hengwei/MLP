import numpy as np

from mlp.utils import Factory, Module


class Activation(Module, Factory):
    def activation(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def derivative(self) -> np.ndarray:
        raise NotImplementedError
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = self.activation(x)
        return self.cache
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        return d * self.derivative()


class Sigmoid(Activation):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self) -> np.ndarray:
        return self.cache * (1 - self.cache)


class Tanh(Activation):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
    
    def derivative(self) -> np.ndarray:
        return 1 - self.cache ** 2
    
    
class ReLU(Activation):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self) -> np.ndarray:
        return (self.cache > 0).astype(int)
    
    
class Linear(Activation):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self) -> np.ndarray:
        return np.ones_like(self.cache)
