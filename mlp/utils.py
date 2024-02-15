import numpy as np

class Factory:
    subclasses = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__.lower()] = cls
    
    @classmethod
    def parse(cls, name: str) -> 'Factory':
        if name.lower() not in cls.subclasses:
            raise ValueError(f'Unknown {cls.__name__}: {name}')
        return cls.subclasses[name.lower()]


class Module:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError
