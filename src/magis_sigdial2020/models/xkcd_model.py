import argparse
import os
import yaml

from magis_sigdial2020.modules.encoders import MLPEncoder
from magis_sigdial2020.utils.model import reload_trial_model
from magis_sigdial2020.utils.nn import new_parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch_safe_log = lambda x: torch.log(torch.clamp(x, 1e-20, 1e7))


class XKCDModel(nn.Module):
    MODEL_TYPE = 'semantic'
    
    @classmethod
    def from_pretrained(cls, dirpath):
        with open(os.path.join(dirpath, 'hparams.yaml')) as fp:
            hparams = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))
        hparams.trial_path = dirpath
        return cls.make(hparams, reload=True, eval_mode=True)
    
    @classmethod
    def make(cls, hparams, reload=False, eval_mode=False):
        model = cls(
            input_size=hparams.input_size,
            encoder_size=hparams.encoder_size,
            encoder_depth=hparams.encoder_depth,
            prediction_size=hparams.prediction_size
        )
        if reload:
            reload_trial_model(model, hparams.trial_path)
        if eval_mode:
            model = model.eval()
        return model
    
    def __init__(self, input_size, encoder_size, encoder_depth, prediction_size):
        super(XKCDModel, self).__init__()
        self.encoder = MLPEncoder(size_in=input_size,
                                  layer_sizes=[encoder_size]*encoder_depth,
                                  add_final_nonlinearity=True)
        self.decoder = nn.Linear(in_features=encoder_size, out_features=prediction_size)
        self.availabilities = new_parameter(1, prediction_size)
    
    def forward(self, x_input):
        output = {}
        
        x_encoded = self.encoder(x_input)
        output['phi_logit'] = self.decoder(x_encoded)
        
        output['log_word_score'] = (
            torch_safe_log(torch.sigmoid(output['phi_logit'])) 
            + torch.log(torch.sigmoid(self.availabilities))
        )
        
        output['word_score'] = torch.exp(output['log_word_score'])
        output['S0_probability'] = F.softmax(output['log_word_score'], dim=1)

        return output
    
class XKCDModelWithRGC(XKCDModel):
    """
    XKCD Model with Referential Goal Composition (RGC). RGC is the algorithm for
    combining semantic computations to compute the probability of a target referent
    and none of the distractors.
    
    In the single object case, this model will behave exactly as the base XKCDModel.
    When alternate objects as passed into the model's call as additional arguments,
    the first is considered the target and the remainder are considered distractors.
    
    Example:
        
        modelb = XKCDModelWithRGC.from_pretrained(pretrained_dir_path)
        x0, x1, x2 = Context.from_cic_row(cic, 0)
    """
    def forward(self, x_target, *x_alts):
        x_target_encoded = self.encoder(x_target)
        x_alts_encoded = list(map(self.encoder, x_alts))
        
        phi_target_logit = self.decoder(x_target_encoded)
        phi_target = torch.sigmoid(phi_target_logit)
        
        if len(x_alts) > 0:
            # shape=(num_alts, batch_size, 829)
            phi_alt = torch.stack([
                torch.sigmoid(self.decoder(x_alt_i_encoded))
                for x_alt_i_encoded in x_alts_encoded
            ])
            phi_alt, _ = torch.max(phi_alt, dim=0)
            # This operation can be understood in multiple ways (tnorms, scales, cdf)
            # The interpretation I am choosing is set restriction on the
            # CDF: P(alt < T < target) = P(T < target) - P(T < alt)
            # The relu is there to guarantee positiveness (obviously if target < alt, the CDF is 0)
            psi_value = F.relu(phi_target - phi_alt)
        else:
            phi_alt = torch.zeros_like(phi_target)
            psi_value = phi_target
            
        word_score = psi_value * torch.sigmoid(self.availabilities)
        S0_probability = word_score / word_score.sum(dim=1, keepdim=True)
            
        return {
            'phi_logit': phi_target_logit,
            'phi_target': phi_target,
            'phi_alt': phi_alt,
            'psi_value': psi_value,
            'word_score': word_score,
            'S0_probability': S0_probability
        }

#Could not find the LSTM model in standfordnlp/color-describer repo
class CompositionalModel(nn.Module):
    MODEL_TYPE = 'semantic'
    
    @classmethod
    def make(cls, hparams, reload=False, eval_mode=False):
        model = cls(
            input_size=hparams.input_size,
            lstm_size=hparams.lstm_size,
            num_lstm_layers=hparams.num_lstm_layers,
            embedding_dim=hparams.embedding_dim,
            vocab_size=hparams.vocab_size,
            max_seq_len=hparams.max_seq_len
        )
        if reload:
            reload_trial_model(model, hparams.trial_path)
        if eval_mode:
            model = model.eval()
        return model
    
    
    #input_size is the size of the transformed color vector (like it is in XKCDModel)
    def __init__(self, input_size, lstm_size, num_lstm_layers, embedding_dim, vocab_size, max_seq_len):
        super(CompositionalModel, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.num_lstm_layers = num_lstm_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size = self.input_size + self.embedding_dim, 
            hidden_size = self.lstm_size, 
            num_layers = self.num_lstm_layers, 
            dropout = 0.2,
            batch_first = True)
        self.fc = nn.Linear(self.lstm_size, self.vocab_size) #!is the input size really lstm_size? i.e. is the output of the lstm lstm_size?
    
    #Teacher forcing: all correct color words inputted during training, not taken from previous step
    #https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    def forward(self, x_input, y_color_name, in_state = None):
        output = {}
        
        embedded = self.embedding(y_color_name)
        
        #expand x_input and concat to embedded y_color_name
            #embedded shape (batch_size, seq_len, embedding_dim)
            #x_input shape (batch_size, input_size) --> (batch_size, seq_len, input_size)
            #concatenated shape (batch_size, seq_len, input_size + embedding_dim)
        x_input = torch.unsqueeze(x_input,1)
        x_input = x_input.expand(-1, self.max_seq_len, -1)
        lstm_input = torch.cat((embedded, x_input), -1)

        #pass through LSTM and FC layers to get logits
        if in_state is None:
            lstm_output, out_state = self.lstm(lstm_input)
        else:
            lstm_output, out_state = self.lstm(lstm_input, in_state)
        logits = self.fc(lstm_output)

        output['logits'] = logits
        output['log_word_score'] = (
            torch_safe_log(torch.sigmoid(output['logits'])) 
        )
        output['word_score'] = torch.exp(output['log_word_score'])
        output['S0_probability'] = F.softmax(output['log_word_score'], dim=2)
        output['probability'] = F.softmax(output['word_score'], dim=2)
        output['state'] = out_state

        return output

class CompositionalXKCDModel(nn.Module):
    MODEL_TYPE = 'semantic'
    
    @classmethod
    def make(cls, hparams, reload=False, eval_mode=False):
        model = cls(
            input_size=hparams.input_size,
            lstm_size=hparams.lstm_size,
            num_lstm_layers=hparams.num_lstm_layers,
            embedding_dim=hparams.embedding_dim,
            vocab_size=hparams.vocab_size,
            max_seq_len=hparams.max_seq_len
        )
        if reload:
            reload_trial_model(model, hparams.trial_path)
        if eval_mode:
            model = model.eval()
        return model
    
    
    #input_size is the size of the transformed color vector (like it is in XKCDModel)
    def __init__(self, input_size, lstm_size, num_lstm_layers, embedding_dim, vocab_size, max_seq_len):
        super(CompositionalXKCDModel, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.num_lstm_layers = num_lstm_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size = self.input_size + self.embedding_dim, 
            hidden_size = self.lstm_size, 
            num_layers = self.num_lstm_layers, 
            dropout = 0.2,
            batch_first = True)
        self.phi_fc = nn.Linear(self.lstm_size, self.vocab_size)
        self.alpha_fc = nn.Linear(self.lstm_size, self.vocab_size)

    #Teacher forcing: all correct color words inputted during training, not taken from previous step
    #https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    def forward(self, x_input, y_color_name, in_state = None):
        output = {}
        
        embedded = self.embedding(y_color_name)
        
        #expand x_input and concat to embedded y_color_name
            #embedded shape (batch_size, seq_len, embedding_dim)
            #x_input shape (batch_size, input_size) --> (batch_size, seq_len, input_size)
            #concatenated shape (batch_size, seq_len, input_size + embedding_dim)
        x_input = torch.unsqueeze(x_input,1)
        x_input = x_input.expand(-1, self.max_seq_len, -1)
        lstm_input = torch.cat((embedded, x_input), -1)

        #pass through LSTM and FC layers to get logits and alpha
        if in_state is None:
            lstm_output, out_state = self.lstm(lstm_input)
        else:
            lstm_output, out_state = self.lstm(lstm_input, in_state)
        phi_logit = self.phi_fc(lstm_output)
        alpha = self.alpha_fc(lstm_output)

        output['phi_logit'] = phi_logit
        output['alpha'] = alpha
        output['log_word_score'] = (
            torch_safe_log(torch.sigmoid(output['phi_logit']))
                        + torch.log(torch.sigmoid(output['alpha']))
        )
        output['word_score'] = torch.exp(output['log_word_score'])
        output['S0_probability'] = F.softmax(output['log_word_score'], dim=2)
        output['state'] = out_state

        return output