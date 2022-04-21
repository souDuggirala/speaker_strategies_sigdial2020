from magis_sigdial2020.datasets.xkcd import XKCD, TeacherGuidedXKCD
from magis_sigdial2020.datasets.xkcd.vectorized import CompositionalXKCD #not sure why XKCD and TeacherGuidedXKCD are able to be accessed through xkcd
from magis_sigdial2020.metrics import compute_accuracy, compute_perplexity
from magis_sigdial2020.models.xkcd_model import XKCDModel, CompositionalXKCDModel
from magis_sigdial2020.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CalibratedXKCDModelTrainer(BaseTrainer):
    def make_dataset(self):
        dataset_exists = self.dataset is not None
        
        # in the case of hyper param search; make sure same calibrated phis
        if dataset_exists:
            current_path = self.dataset.get_teacher_phi_path()
            hparams_path = self.hparams.teacher_phi_path
            is_same_calibrations = (current_path == hparams_path)
        else:
            is_same_calibrations = False
            
        if dataset_exists and is_same_calibrations:
            dataset = self.dataset
        else:
            dataset = TeacherGuidedXKCD(
                teacher_phi_path=self.hparams.teacher_phi_path,
                xkcd_coordinate_system=self.hparams.xkcd_coordinate_system
            )
        
        self.hparams.input_size = dataset.xkcd[0]['x_color_value'].shape[0]
        self.hparams.prediction_size = len(dataset.xkcd.color_vocab)
        
        return dataset

    def make_model(self):
        return XKCDModel(
            input_size=self.hparams.input_size,
            encoder_size=self.hparams.encoder_size,
            encoder_depth=self.hparams.encoder_depth,
            prediction_size=self.hparams.prediction_size
        )

    def setup_loss(self):
        self._ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self._bce_logit_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def compute_model_output(self, batch_dict):
        return self.model(batch_dict['x_color_value'])

    def compute_loss(self, batch_dict, model_output):
        loss = 0
        if self.hparams.use_ce:
            loss += self.hparams.ce_weight * self._ce_loss(model_output['log_word_score'],
                                                        batch_dict['y_color_name'])
        if self.hparams.use_teacher_phi:
            loss += self.hparams.teacher_phi_weight * self._bce_logit_loss(model_output['phi_logit'],
                                                                        batch_dict['teacher_phi'])
        return loss

    def compute_metrics(self, batch_dict, model_output):
        return {
            "perplexity": compute_perplexity(model_output['log_word_score'],
                                             batch_dict['y_color_name'],
                                             True),
            "accuracy": compute_accuracy(model_output['log_word_score'],
                                         batch_dict['y_color_name'])
        }
    
    def get_epochbar_stats(self, train_state):
        return {
            'val_loss': f'{train_state["val.loss"]:0.3f}',
            'val_ppl': f'{train_state["val.perplexity"]:0.3f}'
        }
    

class UncalibratedXKCDModelTrainer(BaseTrainer):
    def make_dataset(self):
        dataset = XKCD.from_settings(coordinate_system=self.hparams.xkcd_coordinate_system)
        self.hparams.input_size = dataset[0]['x_color_value'].shape[0]
        self.hparams.prediction_size = len(dataset.color_vocab)
        return dataset

    def make_model(self):
        return XKCDModel(
            input_size=self.hparams.input_size,
            encoder_size=self.hparams.encoder_size,
            encoder_depth=self.hparams.encoder_depth,
            prediction_size=self.hparams.prediction_size
        )

    def setup_loss(self):
        self._ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def compute_model_output(self, batch_dict):
        return self.model(batch_dict['x_color_value'])

    def compute_loss(self, batch_dict, model_output):
        return self._ce_loss(model_output['log_word_score'],
                             batch_dict['y_color_name'])

    def compute_metrics(self, batch_dict, model_output):
        return {
            "perplexity": compute_perplexity(model_output['log_word_score'],
                                             batch_dict['y_color_name'],
                                             True),
            "accuracy": compute_accuracy(model_output['log_word_score'],
                                         batch_dict['y_color_name'])
        }

class CompositionalXKCDModelTrainer(BaseTrainer):

    def make_dataset(self):
        dataset = CompositionalXKCD.from_settings(coordinate_system=self.hparams.xkcd_coordinate_system, max_seq_len=self.hparams.max_seq_len)
        self.hparams.input_size = dataset[0]['x_color_value'].shape[0]
        self.hparams.vocab_size = len(dataset.color_vocab) + 1 #one more for the padding idx 0
        return dataset

    def make_model(self):
        return CompositionalXKCDModel(
            input_size=self.hparams.input_size,
            lstm_size=self.hparams.lstm_size,
            num_lstm_layers=self.hparams.num_lstm_layers,
            embedding_dim=self.hparams.embedding_dim,
            vocab_size=self.hparams.vocab_size,
            max_seq_len=self.hparams.max_seq_len
        )

    def setup_loss(self):
        self._ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def compute_model_output(self, batch_dict):
        return self.model(batch_dict['x_color_value'], batch_dict['y_color_name'])

    '''later incorporate VAE regularization'''
    def compute_loss(self, batch_dict, model_output):
        target = batch_dict['y_color_name'][:, 1:] #crop out start token
        pred = model_output['log_word_score'][:, :-1, :] #make shape match target
        pred = torch.transpose(pred,1,2) #needs to be shape (batch_size, num_class, other_dim)
        return self._ce_loss(pred, target)

    def compute_metrics(self, batch_dict, model_output):
        target = batch_dict['y_color_name'][:, 1:] #crop out start token
        pred = model_output['log_word_score'][:, :-1, :] #make shape match target
        return {
            "accuracy": compute_accuracy(pred,target, True),
            #perplexity = exp(cross entropy loss) -- https://en.wikipedia.org/wiki/Perplexity 
            "perplexity": torch.exp(self.compute_loss(batch_dict, model_output)).item()
        }