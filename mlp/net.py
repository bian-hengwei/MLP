import numpy as np

from mlp.embedding import Embedding
from mlp.layer import Layer
from mlp.utils import Module
from typing import List


class Net(Module):
    def __init__(self, layers: List[Layer], embeddings: List[Embedding]):
        self.layers = layers
        self.embeddings = embeddings
        
    def __repr__(self) -> str:
        return f'Net({self.layers})'
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        embeddings = [embedding(x) for embedding in self.embeddings]
        x = np.column_stack([x] + embeddings)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d
    
    @classmethod
    def from_config(cls, config: dict) -> 'Net':
        """Assertions:  
        - config['input_dim'] is an integer
        - config['output_dim'] is an integer
        - config['hidden_dims'] is a list of integers
        - config['activations'] is a string or a list of strings with length len(config['hidden_dims']) + 1
        - Each string in config['activations'] is a valid activation function
        """
        layers = []
        
        dims = [config['input_dim']] + config['hidden_dims'] + [config['output_dim']]
        
        activations = config['activations']
        if isinstance(activations, str):
            activations = [activations] * (len(dims) - 1)
            
        for i in range(len(dims) - 1):
            layers.append(Layer(dims[i], dims[i + 1], activations[i], config.get('initialization', 'xavier')))

        embeddings = [Embedding.parse(embedding)() for embedding in config.get('embeddings', [])]
            
        return cls(layers, embeddings)
        