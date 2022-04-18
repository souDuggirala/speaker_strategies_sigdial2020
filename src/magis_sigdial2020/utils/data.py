import colorsys
from collections import deque

import numpy as np
import torch
import torch.utils.data


def generate_batches(dataset, batch_size, split='train', shuffle=True,
                     num_workers=0, pin_memory=False, drop_last=True,
                     device="cpu"):

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, pin_memory=pin_memory,
                            drop_last=drop_last,
                            num_workers=num_workers)

    for batch_index, data_dict in enumerate(dataloader, 1):
        data_dict['batch_index'] = torch.FloatTensor([batch_index])
        yield {name: tensor.to(device) for name, tensor in data_dict.items()}
        
        
class Dataset(torch.utils.data.Dataset):
    def generate_batches(self, batch_size, split='train', shuffle=True,
                         num_workers=0, pin_memory=False, drop_last=True,
                         device="cpu"):
        return generate_batches(
            dataset=self,
            batch_size=batch_size,
            split=split,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            device=device
        )


class Context:
    def __init__(self, object_features, target_index=0, batch_dict=None, utterance_index=None, 
                 full_predictions=None, utterance_predictions=None, cic=None, target_prior=None):
        self.object_features = object_features
        self.target_index = target_index
        if utterance_index is None:
            utterance_index = torch.ones(object_features.shape[0]) * -1
        self.utterance_index = utterance_index
        self.full_predictions = full_predictions
        self.utterance_predictions = utterance_predictions
        self.batch_dict = batch_dict
        self.cic = cic
        self.target_prior = target_prior
        if self.target_prior is None:
            self.reset_target_prior()
        self.prior_history = []
       
    @classmethod
    def from_cic(cls, cic, batch_size=4, deterministic=True, iteration=1):
        generator = generate_batches(cic, shuffle=(not deterministic), batch_size=batch_size)
        for _ in range(iteration):
            batch = next(generator)
        return cls.from_cic_batch(batch, cic=cic)

    @classmethod
    def from_cic_batch(cls, cic_batch, cic=None):
        return cls(object_features=cic_batch['x_colors'].float(), 
                   target_index=torch.zeros_like(cic_batch['y_utterance']).long(),
                   utterance_index=cic_batch['y_utterance'].long(),
                   batch_dict=cic_batch, 
                   cic=cic)

    @classmethod
    def from_cic_row_indices(cls, cic, cic_row_indices):
        batch_dict = {'x_colors': [], 'y_utterance': [], 'row_index': []}
        for row_index in cic_row_indices:
            batch_dict['x_colors'].append(
                cic.x_color_values[[
                    row_index,
                    row_index + cic._offset,
                    row_index + 2 * cic._offset
                ]]
            )
            batch_dict['y_utterance'].append(cic.label_indices[row_index])
            batch_dict['row_index'].append(row_index)
        batch_dict['x_colors'] = torch.FloatTensor(batch_dict['x_colors'])
        batch_dict['y_utterance'] = torch.LongTensor(batch_dict['y_utterance'])
        batch_dict['row_index'] = torch.LongTensor(batch_dict['row_index'])
        return cls.from_cic_batch(batch_dict, cic)
    
    def copy(self):
        cls = self.__class__
        return cls(object_features=self.object_features, 
                   target_index=self.target_index,
                   utterance_index=self.utterance_index,
                   batch_dict=self.batch_dict, 
                   cic=self.cic,
                   target_prior=self.target_prior)
    
    def reset_target_prior(self):
        num_batches, num_objects, _ = self.object_features.size()
        self.target_prior = (
            (torch.ones((num_batches, num_objects)) / num_objects)
            .to(self.object_features.device)
        )
        self.prior_history = []
        
    def update_target_prior(self, new_prior):
        self.prior_history.append(self.target_prior.detach())
        self.target_prior = new_prior
        

    def iterate_reference_sets(self):
        _, num_obj, _ = self.object_features.shape

        object_indices = deque(range(num_obj))
        for _ in range(num_obj):
            yield tuple(self.object_features[:, obj_i] for obj_i in object_indices)
            object_indices.rotate(-1)
            
    def plot(self, target=0, axes=None):
        from magis_sigdial2020.utils.plot import plot_three
        if self.cic is None:
            raise Exception("Set cic first")
        if target >= len(self.batch_dict['row_index']):
            raise Exception(
                f"only {len(self.bach_dict['row_index'])} options, but "
                f"target={target}")
        
        if self.utterance_index.min() >= 0:
            label = self.cic._color_vocab.lookup_index(self.utterance_index[target].item())
        else:
            label = "--"
            
        row_index = self.batch_dict['row_index'][target].item()
        # self.cic._x_color_values is in hsv space
        x0 = self.cic._x_color_values[row_index]
        x1 = self.cic._x_color_values[row_index + self.cic._offset]
        x2 = self.cic._x_color_values[row_index + 2 * self.cic._offset]
        plot_three(x0, x1, x2, label, color_space='hsv', axes=axes)
