import numpy as np

from mlp.numpyNN import sample_data
from typing import Tuple


class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self) -> int:
        return len(self.x) // self.batch_size
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        return self.x[start:end], self.y[start:end]
    
    def __iter__(self) -> 'Dataset':
        if self.shuffle:
            self.shuffle_data()
        self.i = 0
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.i >= len(self):
            raise StopIteration
        self.i += 1
        return self[self.i - 1]
    
    def shuffle_data(self):
        indices = np.random.permutation(len(self.x))
        self.x = self.x[indices]
        self.y = self.y[indices]
    
    @classmethod
    def from_config(cls, config: dict) -> Tuple['Dataset', 'Dataset']:
        x_train, y_train, x_test, y_test = sample_data(config['data_name'], config.get('nTrain', 200), config.get('nTest', 200), config.get('random_seed', 0))
        return cls(x_train, y_train, config['batch_size'], True), cls(x_test, y_test, config['batch_size'], False)
