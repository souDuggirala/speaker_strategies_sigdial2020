import argparse
import gc
import os
from pathlib import Path

from magis_sigdial2020.datasets.xkcd.vectorized import CompositionalXKCD
from magis_sigdial2020.models.xkcd_model import CompositionalModel
from magis_sigdial2020.hyper_params import HyperParameters
import numpy as np
import pyromancy
import pyromancy.subscribers as sub
from pyromancy import reader
from pyromancy.utils import get_args
import torch
import tqdm

BY_SEQ_POSITION = True

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
    
    dataset = CompositionalXKCD.from_settings(coordinate_system=hparams.xkcd_coordinate_system)
    dataset.set_split("train")
    
    model = CompositionalModel.make(
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
                torch.sigmoid(model(batch['x_color_value'], batch["y_color_name"])['logits'])
                .detach().cpu().numpy().astype(np.float32) 
            )
            bar.update(teacher_phi[-1].shape[0])
        teacher_phi = np.concatenate(teacher_phi).astype(np.float32)
        assert teacher_phi.shape == (len(dataset), model.max_seq_len, len(dataset.color_vocab)+1)
    
        np.save(hparams.unnormalized_filepath, teacher_phi)
        print("unnormalized cached")

    print("Beginning normalization")
    gc.collect()
    
    # TODO: if this is too much; do it col by col. 
    if BY_SEQ_POSITION:
        #Compute percentiles separately for each position in sequence
        by_sequence_position = np.transpose(teacher_phi, (1,0,2)) #switch the batch and seq len dims
        percentiles = np.percentile(by_sequence_position, hparams.normalizing_percentile, axis=1)
        print("percentiles computed")
        gc.collect()
    else:
        # Stack everything and take percentiles over all words from sequences
        # np.percentile(np.resize(a,(4,3)), 90, axis = 0)
        stacked = np.resize(teacher_phi, (teacher_phi.shape[0]+teacher_phi.shape[1], teacher_phi.shape[2]))
        percentiles = np.percentile(stacked, hparams.normalizing_percentile, axis=0) 
        print("percentiles computed")
        gc.collect()

    '''
    Division works the way I would want it to in BY_SEQ_POSITION case
    >>> a
    array([[[10,  7,  4],
            [ 3,  2,  1]],

            [[ 9,  6,  3],
            [ 2,  1,  0]]])
    >>> np.percentile(a, 90, axis = 1)
    array([[9.3, 6.5, 3.7],
           [8.3, 5.5, 2.7]]) #in the real implementation, shape[0] = max_seq_len
    >>> percentiles2 = np.percentile(a, 90, axis = 1)
    >>> a/percentiles2
    array([[[1.07526882, 1.07692308, 1.08108108],
            [0.36144578, 0.36363636, 0.37037037]],

            [[0.96774194, 0.92307692, 0.81081081],
            [0.24096386, 0.18181818, 0.        ]]]
    '''

    normalized_teacher_phi = teacher_phi / percentiles
    del teacher_phi
    gc.collect()
    print(f"Normalized; max is {normalized_teacher_phi.max()}")

    normalized_teacher_phi = np.clip(normalized_teacher_phi, 0, 1)
    print(f"Clipped; max is {normalized_teacher_phi.max()}")
    
    np.save(hparams.normalized_filepath, normalized_teacher_phi)
    print("Normalized cached")

if __name__ == "__main__":
    main()