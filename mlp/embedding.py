import numpy as np

from mlp.utils import Factory

class Embedding(Factory):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class x1squared(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0] ** 2


class x2squared(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[:, 1] ** 2


class x1x2(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0] * x[:, 1]


class sinx1(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 0])
    

class sinx2(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 1])
    

class x1squaredplusx2squared(Embedding):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0] ** 2 + x[:, 1] ** 2
