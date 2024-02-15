import numpy as np

from mlp.utils import Factory


class Loss(Factory):
    def __init__(self, net):
        self.net = net
        self.cache = None
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(x, y)
    
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


class L2Loss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.cache = x, y
        return np.mean((x - y) ** 2)
    
    def backward(self):
        x, y = self.cache
        d = 2 * (x - y) / x.shape[0]
        self.net.backward(d)


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.cache = x, y
        x = np.clip(x, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
        return loss
    
    def backward(self):
        x, y = self.cache
        x = np.clip(x, 1e-15, 1 - 1e-15)
        d = -(y / x) + ((1 - y) / (1 - x))
        d /= x.shape[0]
        self.net.backward(d)
