""" The train script. """

import config
from modellib.trainer import Trainer
from dataloader import generate_data


if __name__ == '__main__':
    # 1. Generate config.
    cfg = config.ConfigTrain()
    # 2. Generate trainer & data.
    model = Trainer(cfg)
    train_data = generate_data(cfg)
    # 3. Train
    model.train_model(train_data=train_data)
