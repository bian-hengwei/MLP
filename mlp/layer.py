import numpy as np

from mlp.activation import Activation
from mlp.utils import Module
    
    
class Layer(Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str, initialization: str):
        self.activation = Activation.parse(activation)()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialize(initialization)
        
    def __repr__(self) -> str:
        return f'({self.w.shape[0]}, {self.w.shape[1]}, {self.activation.__class__.__name__})'
    
    def initialize(self, mode: str):
        if mode == 'xavier':
            a = np.sqrt(6 / (self.input_dim + self.output_dim))
        elif mode == 'he':
            a = np.sqrt(2 / self.input_dim)
        else:
            raise ValueError(f'Unknown initialization mode: {mode}')
        
        self.w = np.random.uniform(-a, a, (self.input_dim, self.output_dim))
        self.b = np.zeros(self.output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return self.activation(x @ self.w + self.b)
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        d = self.activation.backward(d)
        self.dw = self.cache.T @ d
        self.db = d.sum(axis=0)
        return d @ self.w.T
