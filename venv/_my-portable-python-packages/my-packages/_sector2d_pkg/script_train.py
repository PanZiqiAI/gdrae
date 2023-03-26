
from _sector2d_trainer.config import ConfigTrainModel
from _sector2d_trainer.f_trainer import Trainer


if __name__ == '__main__':
    # 1. Generate config.
    cfg = ConfigTrainModel()
    # 2. Generate model.
    model = Trainer(cfg=cfg)
    # 3. Train
    model.train_model()
