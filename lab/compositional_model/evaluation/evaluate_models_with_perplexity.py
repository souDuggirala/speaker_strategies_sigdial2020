import argparse
import json
import os
import pathlib
import yaml
import gc
import time

from magis_sigdial2020.models.xkcd_model import CompositionalModel, CompositionalXKCDModel
from magis_sigdial2020.hyper_params import HyperParameters
from magis_sigdial2020.datasets.xkcd.vectorized import CompositionalXKCD
from magis_sigdial2020.metrics import compute_perplexity_seq
from magis_sigdial2020 import settings
import pyromancy
from pyromancy.utils import get_args
import pyromancy.reader as reader
import pyromancy.subscribers as sub
import numpy as np
import pandas as pd
import torch
import tqdm

to_numpy = lambda x: x.cpu().detach().numpy()

RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": pathlib.Path(__file__).absolute().parents[1],
    "LAB_ROOT": f"{settings.REPO_ROOT}/lab"
}


def get_specific_args(exp_name, trial_name):
    exp = reader.SingleExperimentReader(exp_name)
    trial_map = {os.path.split(trial_path)[1]: trial_path for trial_path in exp.all_trial_paths}
    args = get_args(trial_map[trial_name])
    args.trial_path = trial_map[trial_name]
    return args

def parse_hparams():
    """Input a YAML file 
    
    YAML format:
        
        # experiment settings
        experiment_name: E004_evaluate_on_xkcd
        trial_name: published_version
        root_output_path: "{LAB_SUBDIR_ROOT}/logs"
        device: cuda
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("YAML_CONFIG", help='')
    hparams = HyperParameters.load(parser.parse_args().YAML_CONFIG, RUNTIME_INFO)
    return hparams

def instantiate_models(hparams):
    
    compositional_model = CompositionalModel.make(
        get_specific_args(hparams.target_compositional_experiment, hparams.target_compositional_trial),
        reload=True, eval_mode=True
    )
    compositional_xkcd_model = CompositionalXKCDModel.make(
        get_specific_args(hparams.target_compositional_xkcd_experiment, hparams.target_compositional_xkcd_trial),
        reload=True, eval_mode=True
    )
    compositional_model.max_seq_len = 1
    compositional_xkcd_model.max_seq_len = 1
    #load Monroe compositional model
    #load XKCD compositional model
    return compositional_model, compositional_xkcd_model

#target_description contains both start and end token, so output should too
#pred = model_output['log_word_score'][:, :-1, :] #make shape match target
def evaluate_model(model, color_patch, hparams):
    #fill shape (batch_size, 1) with start token
    color_patch = color_patch.to(hparams.device)
    start_token = torch.full((color_patch.size(dim=0), 1),1).long()
    #end_token = torch.full((color_patch.size(dim=0), 1),2).long()

    output = []

    word = start_token
    state = None
    for i in range(hparams.max_seq_len): #a loop that goes on for 6 instead
        word = word.to(hparams.device)
        word_output = model(color_patch, word, state)

        #in the compositional model, this probability is based on output of recurrent step
        #in the compositional xkcd model, this probability is based on phi*alpha for each recurrent step
        #word = torch.from_numpy(np.random.choice(model.vocab_size, p=to_numpy(word_output['S0_probability'])))
        word = torch.argmax(word_output['S0_probability'], axis = -1)
        state = word_output['state']

        output.append(to_numpy(word_output['log_word_score']))

    output = np.array(output).squeeze(axis=2)
    output = np.transpose(output, (1,0,2))
    output= torch.from_numpy(output)

    return output

def main():
    hparams = parse_hparams()

    #need to create hparams yaml file
    pyromancy.settings.set_root_output_path(hparams.root_output_path)
    exp = pyromancy.initialize(
        experiment_name=hparams.experiment_name,
        subscribers=[sub.DBSubscriber()],
        trial_name=hparams.trial_name
    )
    exp.log_exp_start()

    hparams.results_filepath = exp.expand_to_trial_path("results.csv")
    hparams.hparams_filepath = exp.expand_to_trial_path("hparams.yaml")
    hparams.save(hparams.hparams_filepath)

    dataset = CompositionalXKCD.from_settings(coordinate_system=hparams.xkcd_coordinate_system, max_seq_len=6)
    
    results_df = []
    for split in ['train', 'val', 'test']:
        dataset.set_split(split)

        results_df_i = {
            "x_color_value": [], 
            "y_color_name": [],
            "Monroe perplexity": [],
            "CompositionalXKCD perplexity": [],
            "Monroe output": [],
            "CompositionalXKCD output": []
        }

        monroe_model, compositional_xkcd_model = instantiate_models(hparams)
        monroe_model = monroe_model.to(hparams.device) #don't understand these
        compositional_xkcd_model= compositional_xkcd_model.to(hparams.device) #don't understand these
        
        #generate batches
        batch_bar = tqdm.tqdm(total=len(dataset)//hparams.batch_size, leave=False, position=1)
        batch_generator = dataset.generate_batches(
            batch_size=hparams.batch_size, 
            device=hparams.device, 
            drop_last=False, 
            shuffle=False
        )

        for batch in batch_generator:
            target_description = batch['y_color_name']
            color_patch = batch['x_color_value']

            results_df_i['y_color_name'].append(to_numpy(target_description))
            results_df_i['x_color_value'].append(to_numpy(color_patch))
            
            monroe_output = evaluate_model(monroe_model, color_patch, hparams)
            compositional_xkcd_output = evaluate_model(compositional_xkcd_model, color_patch, hparams)
            results_df_i['Monroe output'].append(to_numpy(monroe_output))
            results_df_i['CompositionalXKCD output'].append(to_numpy(compositional_xkcd_output))

            results_df_i['Monroe perplexity'].append(compute_perplexity_seq(monroe_output[:, :-1, :], target_description[:, 1:]))
            results_df_i['CompositionalXKCD perplexity'].append(compute_perplexity_seq(compositional_xkcd_output[:, :-1, :], target_description[:, 1:]))

            batch_bar.update()

        results_df_i = pd.DataFrame(results_df_i)
        results_df_i['split'] = split
        
        results_df.append(results_df_i)

    results_df = pd.concat(results_df)
    print("Monroe average perplexity: " + str(results_df['Monroe perplexity'].mean()))
    print("CompositionalXKCD average perplexity: " + str(results_df['CompositionalXKCD perplexity'].mean()))
    results_df.to_csv(hparams.results_filepath, index=None)
    exp.log_exp_end()

if __name__ == "__main__":
    with torch.no_grad():
        main()
