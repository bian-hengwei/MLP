import numpy as np

from mlp.layer import Layer
from mlp.utils import Factory
from typing import List

class Optimizer(Factory):
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def step(self):
        raise NotImplementedError
    
    @classmethod
    def build(cls, config: dict, layers: List[Layer]) -> 'Optimizer':
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate: float, decay: float):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
    
    def step(self):
        self.iterations += 1
        lr = self.learning_rate / (1 + self.decay * self.iterations)
        
        for layer in self.layers:
            layer.w -= lr * layer.dw
            layer.b -= lr * layer.db

    @classmethod
    def build(cls, config: dict, layers: List[Layer]) -> 'Optimizer':
        return cls(layers, config['learning_rate'], config.get('decay', 0.0))
            
            
class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate: float, momentum: float):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = [np.zeros_like(layer.w) for layer in layers]
        self.velocity_b = [np.zeros_like(layer.b) for layer in layers]
    
    def step(self):
        for i, layer in enumerate(self.layers):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * layer.dw
            layer.w += self.velocity_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * layer.db
            layer.b += self.velocity_b[i]
    
    @classmethod
    def build(cls, config: dict, layers: List[Layer]) -> 'Optimizer':
        return cls(layers, config['learning_rate'], config.get('momentum', 0.9))
    

class Adam(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate: float, beta_1: float, beta_2: float, epsilon: float):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0
        self.m_w = [np.zeros_like(layer.w) for layer in layers]
        self.v_w = [np.zeros_like(layer.w) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
    
    def step(self):
        self.iterations += 1
        for i, layer in enumerate(self.layers):
            # Update weights
            self.m_w[i] = self.beta_1 * self.m_w[i] + (1 - self.beta_1) * layer.dw
            self.v_w[i] = self.beta_2 * self.v_w[i] + (1 - self.beta_2) * (layer.dw ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta_1 ** self.iterations)
            v_hat_w = self.v_w[i] / (1 - self.beta_2 ** self.iterations)
            layer.w -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            
            # Update biases
            self.m_b[i] = self.beta_1 * self.m_b[i] + (1 - self.beta_1) * layer.db
            self.v_b[i] = self.beta_2 * self.v_b[i] + (1 - self.beta_2) * (layer.db ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta_1 ** self.iterations)
            v_hat_b = self.v_b[i] / (1 - self.beta_2 ** self.iterations)
            layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    @classmethod
    def build(cls, config: dict, layers: List[Layer]) -> 'Optimizer':
        return cls(layers, config['learning_rate'], config.get('beta_1', 0.9), config.get('beta_2', 0.999), config.get('epsilon', 1e-7))
