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
    exp = reader.SingleExperimentReader(exp_name)#, filter_unfinished=False)
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

'''
REUSING SAME LOGIC FOR CUMULATIVE CASE, HAVE TO REWRITE THIS  DOCUMENTATION AND RECHECK THIS LOGIC
Recursively normalize phi in order to condition normalization for each sequence position on all the positions before it.
For sequence position 1, the token conditioned on is always START, so sequence position 1 is normalized across the whole dataset
For sequence position 2, there are some number of unique tokens in preceding position, so the algorithm iterates normalize separately for each unique token
For sequence position 3+, the normalization must be done not only on the immediately preceding unique tokens, but the combinations of all unique tokens preceding
After normalizing sequence position s_n | s_1, ..., s_n-1, recursively normalize conditioned on s_n, to get s_n+1 | s_1, ..., s_n

color_descriptions is input to the model (with START included) and teacher_phi is output of the model
    the rows of both should correspond (because shuffle was set to false in batch_generator)
    the shapes should be the same except for the sequence position axis -- color_descriptions should be 1 greater to include START
'''
def recurse_normalize(teacher, normalizing_percentile, color_descriptions):
    #stop condition is when teacher is empty
    if teacher.shape[1]==0:
        return

    normalized_teacher = np.zeros_like(teacher)
    #find the unique words in the current sequence position, then iterate to normalize conditioned on these unique words
    unique_words = np.unique(color_descriptions[:,0])
    for word in unique_words:
        #choose the samples corresponding to unique word being the input
        indices = np.where(color_descriptions[:,0]==word)[0]
        #normalize over those samples
        #at this step, the shapes are different between teacher_phi and teacher_prob, but it should be doing the same thing
            #teacher_phi[indices,0] shape = (num_samples, vocab_size) ----> percentile_phi (axis = 0) shape = (vocab_size,)
                #calculating a percentile separately for each item in vocabulary
            #teacher_prob[indices,0] shape = (num_samples,) ----> percentile_phi (axis = 0) shape = a single value
        percentile = np.percentile(teacher[indices,0], normalizing_percentile, axis=0)
        #calculate normalized value for the current sequence position and chosen samples
        normalized_teacher[indices,0] = teacher[indices,0]/percentile
        #normalize the next sequence position step recursively, only passing on the chosen samples
        normalized_teacher[indices, 1:] = recurse_normalize(teacher[indices, 1:], normalizing_percentile, color_descriptions[indices, 1:])

    return normalized_teacher 

def main():
    hparams = parse_hparams()
    pyromancy.settings.set_root_output_path(hparams.root_output_path)
    exp = pyromancy.initialize(
        experiment_name=hparams.experiment_name,
        subscribers=[sub.DBSubscriber()],
        trial_name=hparams.trial_name
    )
    
    hparams.unnormalized_filepath = exp.expand_to_trial_path(f"unnormalized_{hparams.teacher_type}.npy")
    hparams.normalized_filepath = exp.expand_to_trial_path(
        f"calibrated_{hparams.teacher_type}_{hparams.normalizing_percentile}percentile.npy"
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
        teacher = np.load(hparams.unnormalized_filepath)
    else:
        print("Unnormalized does not exist; making")
        teacher = []
        model_output_key = 'logits' if hparams.teacher_type == 'phis' else 'Cum_probability'
        bar = tqdm.tqdm(total=len(dataset))
        
        # TODO: assess whether this should be on GPU; or do it not matter?
        batch_generator = dataset.generate_batches(
            batch_size=1024, 
            shuffle=False, 
            drop_last=False
        )
        i = 0
        for batch in batch_generator:
            teacher.append(
                #not return right shape for probs case: returning (6,6) when it should be returning (1024, 6)
                torch.sigmoid(model(batch['x_color_value'], batch["y_color_name"])[model_output_key])
                .detach().cpu().numpy().astype(np.float32) 
            )
            bar.update(teacher[-1].shape[0])
        teacher = np.concatenate(teacher).astype(np.float32)
        if model_output_key == 'logits':
            assert teacher.shape == (len(dataset), model.max_seq_len, len(dataset.color_vocab)+1)
        else:
            assert teacher.shape == (len(dataset), model.max_seq_len)
    
        np.save(hparams.unnormalized_filepath, teacher)
        print("unnormalized cached")

    print("Beginning normalization")
    gc.collect()
     
    if hparams.normalization_method == 'recursive':
        color_descriptions = np.array([tup[1] for tup in dataset._target_fast])
        normalized_teacher = recurse_normalize(teacher, hparams.normalizing_percentile, color_descriptions)

    elif hparams.normalization_method == 'by_seq_position':
        #Compute percentiles separately for each position in sequence
        percentiles = np.percentile(teacher, hparams.normalizing_percentile, axis=0) #percentile over batch dimension
        print("percentiles computed")
        gc.collect()
        normalized_teacher = teacher / percentiles

    else:
        # Stack everything and take percentiles over all words from sequences
        # np.percentile(np.resize(a,(4,3)), 90, axis = 0)
        stacked = np.resize(teacher, (teacher.shape[0]+teacher.shape[1], teacher.shape[2]))
        percentiles = np.percentile(stacked, hparams.normalizing_percentile, axis=0) 
        print("percentiles computed")
        gc.collect()
        normalized_teacher = teacher / percentiles

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

    assert normalized_teacher.shape == teacher.shape
    del teacher
    gc.collect()

    if hparams.replace_nan_with_0 is True:
        normalized_teacher = np.nan_to_num(normalized_teacher, copy = False)

    print(f"Normalized; max is {normalized_teacher.max()}")

    normalized_teacher = np.clip(normalized_teacher, 0, 1)
    print(f"Clipped; max is {normalized_teacher.max()}")
    
    np.save(hparams.normalized_filepath, normalized_teacher)
    print("Normalized cached")

if __name__ == "__main__":
    main()