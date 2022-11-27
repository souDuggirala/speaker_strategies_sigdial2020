import os
import sys
import pathlib
import argparse
import torch
import numpy as np

from magis_sigdial2020.hyper_params import HyperParameters
import pyromancy
from pyromancy.utils import get_args
import pyromancy.reader as reader
import pyromancy.subscribers as sub
import tqdm

from magis_sigdial2020.datasets.colorspace import get_colorspace
from magis_sigdial2020.datasets.xkcd.vectorized import CompositionalXKCD
from magis_sigdial2020.models.xkcd_model import CompositionalModel, CompositionalXKCDModel
from magis_sigdial2020 import settings


EXPERIMENT = 'E007_CompositionalXKCDModel'
TRIAL = 'trial1-probs'
DEVICE = 'cuda'
COORDINATE_SYSTEM = 'fft'

'''
RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": pathlib.Path(__file__).absolute().parents[1],
    "LAB_ROOT": f"{settings.REPO_ROOT}/lab"
}
'''

to_numpy = lambda x: x.cpu().detach().numpy()

def apply_model_to_colorspace(color_term_indices, model, csd):

    color_term_indices = np.resize(color_term_indices, model.max_seq_len).reshape(1, model.max_seq_len)
    color_term_indices = torch.from_numpy(color_term_indices)

    phi = []
    to_numpy = lambda tensor: tensor.cpu().detach().numpy()

    batch_generator = csd.generate_batches(
        batch_size=256, 
        shuffle=False, 
        drop_last=False,
        device=DEVICE
    )

    for _, batch in enumerate(batch_generator):
        color_term_input = color_term_indices.expand(batch['x_colors'].size(dim=0), -1)
        model_output = model(batch['x_colors'], color_term_input)
        phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))

    phi = np.vstack(phi)
    return phi

def get_specific_args(exp_name, trial_name):
    exp = reader.SingleExperimentReader(exp_name)
    trial_map = {os.path.split(trial_path)[1]: trial_path for trial_path in exp.all_trial_paths}
    args = get_args(trial_map[trial_name])
    args.trial_path = trial_map[trial_name]
    return args

'''
def parse_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("YAML_CONFIG", help='')
    hparams = HyperParameters.load(parser.parse_args().YAML_CONFIG, RUNTIME_INFO)
    return hparams

hparams = parse_hparams()
pyromancy.settings.set_root_output_path(hparams.root_output_path)
exp = pyromancy.initialize(
    experiment_name=hparams.experiment_name,
    subscribers=[sub.DBSubscriber()],
    trial_name=hparams.trial_name
)
exp.log_exp_start()

hparams.output_filepath = exp.expand_to_trial_path("output.txt")
hparams.hparams_filepath = exp.expand_to_trial_path("hparams.yaml")
hparams.save(hparams.hparams_filepath)
sys.stdout = open(hparams.output_filepath, 'w')
'''

model = CompositionalXKCDModel.make(
    get_specific_args(EXPERIMENT, TRIAL),
    reload=True, eval_mode=True)
model.to(DEVICE)

csd = get_colorspace(coordinate_system=COORDINATE_SYSTEM)

dataset = CompositionalXKCD.from_settings(coordinate_system = 'fft')
bar = tqdm.tqdm(total=len(dataset)/256)

batch_generator = dataset.generate_batches(
        batch_size=256, 
        shuffle=False, 
        drop_last=False,
        device=DEVICE
    )
max_phis = []

for batch in batch_generator:
    phi = apply_model_to_colorspace(batch['y_color_name'], model, csd)
    max_phis.append(np.max(phi, axis=0)) #max across colorspace (the previous step is all the values across colorspace)
    #print(max_phis[-1].shape)
    bar.update(1)

max_phis = np.vstack(max_phis)
print(np.mean(max_phis))
