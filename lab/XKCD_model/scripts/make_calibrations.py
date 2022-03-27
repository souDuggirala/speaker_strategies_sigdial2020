import argparse
import gc
import os
from pathlib import Path

from magis_sigdial2020.datasets.xkcd import XKCD
from magis_sigdial2020.models.xkcd_model import XKCDModel
from magis_sigdial2020.hyper_params import HyperParameters
import numpy as np
import pyromancy
import pyromancy.subscribers as sub
from pyromancy import reader
from pyromancy.utils import get_args
import torch
import tqdm

RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": Path(__file__).absolute().parents[1]
}


def format_runtime_strings(hparams):
    for key, value in vars(hparams).items():
        if not isinstance(value, str):
            continue
        if '{' not in value:
            continue
        setattr(hparams, key, value.format(**RUNTIME_INFO))


def get_specific_args(exp_name, trial_name):
    exp = reader.SingleExperimentReader(exp_name)
    trial_map = {os.path.split(trial_path)[1]: trial_path for trial_path in exp.all_trial_paths}
    args = get_args(trial_map[trial_name])
    args.trial_path = trial_map[trial_name]
    return args


def parse_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("YAML_CONFIG", help='')
    hparams = HyperParameters.load(parser.parse_args().YAML_CONFIG)
    format_runtime_strings(hparams)
    return hparams


def main():
    hparams = parse_hparams()
    pyromancy.settings.set_root_output_path(hparams.root_output_path)
    exp = pyromancy.initialize(
        experiment_name=hparams.experiment_name,
        subscribers=[sub.DBSubscriber()],
        trial_name=hparams.trial_name
    )
    
    hparams.unnormalized_filepath = exp.expand_to_trial_path("unnormalized_phis.npy")
    hparams.normalized_filepath = exp.expand_to_trial_path(
        f"calibrated_phis_{hparams.normalizing_percentile}percentile.npy"
    )
    
    dataset = XKCD.from_settings(coordinate_system=hparams.xkcd_coordinate_system)
    dataset.set_split("train")
    
    model = XKCDModel.make(
        get_specific_args(hparams.target_experiment_name, hparams.target_trial_name),
        reload=True, eval_mode=True
    )
    
    if os.path.exists(hparams.normalized_filepath):
        print(f"{hparams.normalized_filepath} exists; exitting")
        return
    
    if os.path.exists(hparams.unnormalized_filepath):
        print("Unnormalized exists; loading")
        teacher_phi = np.load(hparams.unnormalized_filepath)
    else:
        print("Unnormalized does not exist; making")
        teacher_phi = []
        bar = tqdm.tqdm(total=len(dataset))
        
        # TODO: assess whether this should be on GPU; or do it not matter?
        batch_generator = dataset.generate_batches(
            batch_size=1024, 
            shuffle=False, 
            drop_last=False
        )
        for batch in batch_generator:
            teacher_phi.append(
                torch.sigmoid(model(batch['x_color_value'])['phi_logit'])
                .detach().cpu().numpy().astype(np.float32) 
            )
            bar.update(teacher_phi[-1].shape[0])
        teacher_phi = np.concatenate(teacher_phi).astype(np.float32)
        assert teacher_phi.shape == (len(dataset), len(dataset.color_vocab))
    
        np.save(hparams.unnormalized_filepath, teacher_phi)
        print("unnormalized cached")

    print("Beginning normalization")
    gc.collect()

    # TODO: if this is too much; do it col by col. 
    percentiles = np.percentile(teacher_phi, hparams.normalizing_percentile, axis=0)
    print("percentiles computed")
    gc.collect()

    normalized_teacher_phi = teacher_phi / percentiles[None, :]
    del teacher_phi
    gc.collect()
    print(f"Normalized; max is {normalized_teacher_phi.max()}")
    
    normalized_teacher_phi = np.clip(normalized_teacher_phi, 0, 1)
    print(f"Clipped; max is {normalized_teacher_phi.max()}")
    
    np.save(hparams.normalized_filepath, normalized_teacher_phi)
    print("Normalized cached")

if __name__ == "__main__":
    main()