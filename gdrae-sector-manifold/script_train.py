""" The train script. """

import config
from dataloader import generate_data
from modellib.model import Trainer


if __name__ == '__main__':
    # 1. Generate config.
    cfg = config.ConfigTrainModel()
    # 2. Generate model.
    model = Trainer(cfg)
    train_data = generate_data(cfg)
    # 3. Train
    model.train_model(train_data=train_data)
