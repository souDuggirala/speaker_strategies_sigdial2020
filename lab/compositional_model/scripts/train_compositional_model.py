import argparse
import pathlib

from magis_sigdial2020.trainers.xkcd_model_trainers import CompositionalModelTrainer
from magis_sigdial2020.hyper_params import HyperParameters
import tqdm
import torch

RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": pathlib.Path(__file__).absolute().parents[1]
}


def format_runtime_strings(hparams):
    for key, value in vars(hparams).items():
        if not isinstance(value, str):
            continue
        if '{' not in value:
            continue
        setattr(hparams, key, value.format(**RUNTIME_INFO))

def parse_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("YAML_CONFIG", help='')
    hparams = HyperParameters.load(parser.parse_args().YAML_CONFIG)
    
    hparams.cuda = hparams.cuda and torch.cuda.is_available()
    hparams.device = ("cuda" if hparams.cuda else "cpu")
    format_runtime_strings(hparams)
    return hparams


def main():
    hparams = parse_hparams()
    trainer = CompositionalModelTrainer(hparams, bar_func=tqdm.tqdm)
    print(hparams)

    try:
        trainer.run_once()
    except KeyboardInterrupt:
        print("---\n---")
        if input("||| Save Exp? (y/n) > ").lower() in ('y', 'yes'):
            trainer.exp.log_exp_end()
            
if __name__ == "__main__":
    main() 