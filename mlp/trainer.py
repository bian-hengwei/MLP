import numpy as np
import matplotlib.pyplot as plt

from mlp.dataset import Dataset
from mlp.loss import Loss
from mlp.net import Net
from mlp.numpyNN import plot_loss, plot_decision_boundary
from mlp.optimizer import Optimizer


class Trainer:
    def __init__(self, config: dict):
        self.train_dataset, self.test_dataset = Dataset.from_config(config)
        self.net = Net.from_config(config)
        self.loss = Loss.parse(config['loss'])(self.net)
        self.optimizer = Optimizer.parse(config['optimizer']).build(config, self.net.layers)
        self.config = config
        print(self.net)
        
    def train(self):
        log = {'train_loss': list(), 'test_loss': list()}
        for epoch in range(self.config['epochs']):
            train_loss = 0
            for x, y in self.train_dataset:
                y_pred = self.net(x)
                train_loss += self.loss(y_pred, y)
                self.loss.backward()
                self.optimizer.step()
            train_loss /= len(self.train_dataset)
                
            print(f'Epoch {epoch + 1}/{self.config["epochs"]}, Loss: {train_loss:.4f}')
            
            log['train_loss'].append(train_loss)
            log['test_loss'].append(self.test())
            
        print(f'Final test loss: {log["test_loss"][-1]:.4f}')
        self.log = log
        
    def test(self):
        test_loss = 0
        for x, y in self.test_dataset:
            y_pred = self.net(x)
            test_loss += self.loss(y_pred, y)
        test_loss /= len(self.test_dataset)
        return test_loss
    
    def save_or_plot(self, name):
        save_pattern = self.config.get('save_pattern', False)
        if save_pattern:
            plt.savefig(save_pattern.replace('*', name))
        else:
            plt.show()
    
    def plot(self, x, y):
        plot_decision_boundary(x, y, self.net)
        self.save_or_plot('boundary')
        
    def plot_train(self):
        self.plot(self.train_dataset.x, self.train_dataset.y)
    
    def plot_test(self):
        self.plot(self.test_dataset.x, self.test_dataset.y)
        
    def plot_loss(self):
        plot_loss(self.log)
        self.save_or_plot('loss')
