import argparse
import json

from mlp.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    trainer = Trainer(config)
    trainer.train()
    trainer.plot_test()
    trainer.plot_loss()
