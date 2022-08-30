import os
import pathlib
import numpy as np
from scipy.special import softmax, expit
import gc

import pyromancy
from pyromancy.utils import get_args
import pyromancy.reader as reader
import torch

from magis_sigdial2020.models.xkcd_model import CompositionalModel, CompositionalXKCDModel
from magis_sigdial2020.datasets.xkcd.vectorized import CompositionalXKCD
from magis_sigdial2020 import settings

RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": pathlib.Path(__file__).absolute().parents[1],
    "LAB_ROOT": f"{settings.REPO_ROOT}/lab"
}

to_numpy = lambda x: x.cpu().detach().numpy()

DEVICE = 'cpu'

def get_specific_args(exp_name, trial_name):
    exp = reader.SingleExperimentReader(exp_name, filter_unfinished = False)
    trial_map = {os.path.split(trial_path)[1]: trial_path for trial_path in exp.all_trial_paths}
    args = get_args(trial_map[trial_name])
    args.trial_path = trial_map[trial_name]
    return args

compositional_xkcd_model = CompositionalXKCDModel.make(
    get_specific_args('E007_CompositionalXKCDModel', 'trial6'),
    reload=True, eval_mode=True)
compositional_xkcd_model.to(DEVICE)

dataset = CompositionalXKCD.from_settings(coordinate_system = "hue")

for split in ['train', 'val', 'test']:
    dataset.set_split(split)
    print(f"Split {split}")

    batch_generator = dataset.generate_batches(
                batch_size=256, 
                shuffle=False, 
                drop_last=False,
                device=DEVICE)

    #phi_logit = []
    phi = []
    alpha_logit = []

    for _, batch in enumerate(batch_generator):
        model_output = compositional_xkcd_model(batch['x_color_value'], batch['y_color_name'])

        #p_word.append(to_numpy(model_output['S0_probability']))
        #phi_logit.append(to_numpy(model_output['phi_logit']))
        phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))
        alpha_logit.append(to_numpy(model_output['alpha']))

    #p_word = np.vstack(p_word)
    #phi_logit = np.vstack(phi_logit)
    phi = np.vstack(phi)
    alpha_logit = np.vstack(alpha_logit)

    #phi_logit_max_across_colorspace = phi_logit.max(axis=2)
    #print(phi_logit_max_across_colorspace.mean(axis=0)) #[0.9461505 0.6857982 3.0268333 3.7105126 3.3004346 2.5877583]
    #print(expit(phi_logit_max_across_colorspace.mean(axis=0))) #[0.7203403  0.66503155 0.95377177 0.9761193  0.96444374 0.93006957]

    phi_max_across_colorspace = phi.max(axis=2)
    print("Mean phi max across colorspace, by seq position")
    print(phi_max_across_colorspace.mean(axis=0))#[0.71257055 0.6310683  0.8361427  0.94985276 0.9258338  0.91956717]
    print("Max of the phi max across colorspace, by seq position")
    print(phi_max_across_colorspace.max(axis=0))#[0.9535155  0.9997931  0.99999976 0.9999863  0.9999778  0.9997898 ]
    print("Min of the phi max across colorspace, by seq position")
    print(phi_max_across_colorspace.min(axis=0))#[0.44485766 0.05784886 0.07593707 0.72008616 0.7996646  0.8190986 ]

    print("Percent of the time max phi across colorspace is >.90")
    print((phi_max_across_colorspace>0.90).sum()/phi_max_across_colorspace.size) #0.4471171710657638
    print((phi_max_across_colorspace>0.90).sum(axis=0)/phi_max_across_colorspace.shape[0]) #[0.00741628 0.08894007 0.681966   0.71006495 0.55540099 0.63891474]

    #check that alpha is not 0
    print("Mean alpha logit: " + str(alpha_logit.mean()))#-16.54223
    print("Mean alpha: " + str(expit(alpha_logit.mean()))) #6.5433554e-08

    #del p_word
    #del phi_logit
    del phi
    del alpha_logit

    gc.collect()



'''
The performance is not great but not as bad as the scales in plots would suggest
'''

''' unnormalized teacher_phi has already been passed through sigmoid
teacher_phi.append(
torch.sigmoid(model(batch['x_color_value'], batch["y_color_name"])['logits'])
.detach().cpu().numpy().astype(np.float32) 
)'''

''' teacher_phi is left unchanged in TeacherGuidedXKCD training dataset
def __getitem__(self, index):
    output = self.xkcd[index]
    if self.split == "train":
        teacher_phi = self.teacher_phi[index]
    else:
        if self.compositional:
            teacher_phi = np.zeros((self.max_seq_len,self.vocab_size)).astype(np.float32)
            seq_indices = np.arange(self.max_seq_len)
            teacher_phi[seq_indices,output['y_color_name']] = 1 #maybe there'll be something wrong with the data type of y_color_name
        else:
            teacher_phi = np.zeros(self.vocab_size).astype(np.float32)
            teacher_phi[output['y_color_name']] = 1
    output['teacher_phi'] = teacher_phi
    return output
'''

''' loss is between phi logit output (not passed through sigmoid) and teacher phi (normalized phi's which were passed through sigmoid)

def compute_loss(self, batch_dict, model_output):
    loss = 0
    if self.hparams.use_ce:
        loss += self.hparams.ce_weight * self._ce_loss(model_output['log_word_score'],
                                                    batch_dict['y_color_name'])
    if self.hparams.use_teacher_phi:
        loss += self.hparams.teacher_phi_weight * self._bce_logit_loss(model_output['phi_logit'],
                                                                    batch_dict['teacher_phi'])
    return loss

def compute_loss(self, batch_dict, model_output):
    loss = 0
    if self.hparams.use_ce:
        target = batch_dict['y_color_name'][:, 1:] #crop out start token
        pred = model_output['log_word_score'][:, :-1, :] #make shape match target
        pred = torch.transpose(pred,1,2) #needs to be shape (batch_size, num_class, other_dim)
        loss += self.hparams.ce_weight * self._ce_loss(pred, target)
    if self.hparams.use_teacher_phi:
        loss += self.hparams.teacher_phi_weight * self._bce_logit_loss(model_output['phi_logit'],
                                                                    batch_dict['teacher_phi'])
    return loss
'''