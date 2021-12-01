import yaml

from fastspeech_model.dataset import FastSpeech2DataModule

if __name__ == "__main__":
    with open('fastspeech2_train.yaml') as f:
        cfg = yaml.load(f)

    FastSpeech2DataModule(**cfg['dataset'])
