import yaml
from pytorch_lightning import Trainer

from fastspeech_model.dataset import FastSpeech2DataModule
from fastspeech_model.fastspeech2 import FastSpeech2Model

if __name__ == "__main__":
    with open('fastspeech2_train.yaml') as f:
        cfg = yaml.load(f)

    datamodule = FastSpeech2DataModule(**cfg['dataset'])
    model = FastSpeech2Model(**cfg['model'])
    trainer = Trainer()
    trainer.fit(model, datamodule)
