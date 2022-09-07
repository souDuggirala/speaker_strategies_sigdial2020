from itertools import filterfalse
from magis_sigdial2020.datasets.colorspace import get_colorspace
from magis_sigdial2020.datasets.xkcd.vectorized import XKCD, CompositionalXKCD
from magis_sigdial2020.utils import color
from magis_sigdial2020.utils.data import generate_batches
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_one(color_vector, color_space='xy', ax=None, title=""):
    if color_space == 'xy':
        rgb_vector, _ = color.xy2rgbhsv(color_vector)
    elif color_space == 'hsl':
        rgb_vector = color.hsl2rgb(*color_vector)
    elif color_space == 'hsv':
        rgb_vector = color.hsv2rgb(*color_vector)
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.add_patch(plt.Rectangle((0, 0), 20, 20, color=tuple(rgb_vector)))
    ax.axis('off')
    ax.set_title(title)

    
def plot_three(color_vector1, color_vector2, color_vector3, title='', color_space='hsv', axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3)
    else:
        assert len(axes) == 3, "Need axes with 3 subplots"
    plot_one(color_vector1, color_space=color_space, ax=axes[0], title=f'target ({title})')
    plot_one(color_vector2, color_space=color_space, ax=axes[1], title=f'alt ({title})')
    plot_one(color_vector3, color_space=color_space, ax=axes[2], title=f'alt ({title})')
    

def plot_row(row, transcript_title=False, subplot_top=0.8):
    """Plot a row from the CIC data frame
    
    Assumption: 
        The color values in row are still in HSL space
    """
    fig, axes = plt.subplots(1, 3)
    for object_name, ax in zip(['target', 'alt1', 'alt2'], axes):
        plot_one(color_vector=row[object_name], color_space='hsl', ax=ax, title=object_name)
    success_string = "successful" if row.clicked == 'target' else f"failed, clicked {row.clicked}"
    if transcript_title:
        transcript = "\n".join([f"[{i}] {ev['role'][0].upper()}: {ev['text']}" for i, ev in enumerate(row.utterance_events)])
        plt.suptitle(f"TRANSCRIPT [{success_string}]\n----------------- \n{transcript}", ha='left', x=0.1)
    elif isinstance(row.lux_label, str):
        plt.suptitle(f"matches lux label: {row.lux_label}; \nFull={row.utterance_events[0]['text']}")
    else:
        plt.suptitle(f"[pattern={row.utterance_events_pattern}] First is: {row.utterance_events[0]['text']}")
    plt.tight_layout()
    plt.subplots_adjust(top=subplot_top)
    

class ColorspacePlotter:
    def __init__(self, model, coordinate_system="fft", cuda=True):
        self.xkcd = XKCD.from_settings(coordinate_system=coordinate_system)
        self.csd = get_colorspace(coordinate_system=coordinate_system)
        self.device = ("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        
        self._label2rows = {
            i:[] for i in self.xkcd.color_vocab._idx_to_token.keys()
        }
        for row_index, label_index in self.xkcd._target_fast:
            self._label2rows[label_index].append(row_index)
            
        self._label2rows = {
            label_index: np.array(row_indices, dtype=np.int32) 
            for label_index, row_indices in self._label2rows.items()
        }

    def apply_model_to_colorspace(self, eps=None):
        p_word = []
        phi = []
        to_numpy = lambda tensor: tensor.cpu().detach().numpy()
        
        batch_generator = self.csd.generate_batches(
            batch_size=256, 
            shuffle=False, 
            drop_last=False,
            device=self.device
        )
        
        for batch_index, batch in enumerate(batch_generator):
            model_output = self.model(batch['x_colors'])
            p_word.append(to_numpy(model_output['S0_probability']))
            phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))
            
        p_word = np.vstack(p_word)
        p_word = p_word.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, p_word.shape[-1]))
        phi = np.vstack(phi)
        phi = phi.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, phi.shape[-1]))

        return p_word, phi

    def contour_plot(self, color_term, target='p_word', levels=[0.1, 0.5], 
                     linestyles=['--', '-'], figsize=(15, 5), title_prefix='', 
                     dim_reduce_func=np.mean, num_to_scatter=-1, scatter_seed=0):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        p_word, phi = self.apply_model_to_colorspace()
        activations = p_word if target=='p_word' else phi
        if len(title_prefix) > 0:
            title_prefix = title_prefix.strip() + ' '

        index = self.xkcd.color_vocab.lookup_token(color_term)
        
        
        if num_to_scatter > 0:
            random_state = np.random.RandomState(seed=scatter_seed)
            xkcd_row_indices = self._label2rows[index]
            subset = random_state.choice(
                xkcd_row_indices, 
                size=min(num_to_scatter, len(xkcd_row_indices)), 
                replace=False
            )
            full_scatter_points = self.xkcd._original_data_matrix[subset]

        im = axes[0].contourf(self.csd.h, self.csd.s, dim_reduce_func(activations, axis=2)[:, :, index].T, alpha=0.3)
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Hue")
        axes[0].set_ylabel("Saturation")
        if num_to_scatter > 0:
            axes[0].scatter(full_scatter_points[:, 0], full_scatter_points[:, 1], marker='x', color='black', alpha=0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        im = axes[1].contourf(self.csd.h, self.csd.v, dim_reduce_func(activations, axis=1)[:, :, index].T, alpha=0.3)
        plt.colorbar(im, ax=axes[1])
        axes[1].set_xlabel("Hue")
        axes[1].set_ylabel("Value")
        if num_to_scatter > 0:
            axes[1].scatter(full_scatter_points[:, 0], full_scatter_points[:, 2], marker='x', color='black', alpha=0.5)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if target == "phi":
            plt.suptitle(f'Applicability (phi) of {color_term} across colorspace')
        elif target == "p_word":
            plt.suptitle(f'Probability of {color_term} across colorspace')

    def plot_both_contours(self, color_term, **kwargs):
        self.contour_plot(color_term, 'phi', **kwargs)
        self.contour_plot(color_term, 'p_word', **kwargs)

class CompositionalColorspacePlotter(ColorspacePlotter):

    def __init__(self, model, coordinate_system="fft", cuda=True):
        super().__init__(model, coordinate_system, cuda)
        self.comp_xkcd = CompositionalXKCD.from_settings(coordinate_system=coordinate_system)

        self._seqword2rows = {
            (word_index, seq_position):([],[]) 
            for word_index in self.comp_xkcd.color_vocab._idx_to_token.keys() 
            for seq_position in range(self.comp_xkcd.max_seq_len)
        }
        
        for target_index in range(len(self.comp_xkcd._target_fast)):
            row_index, label_indices = self.comp_xkcd._target_fast[target_index]
            for seq_position in range(len(label_indices)):     
                word_index = label_indices[seq_position]
                if word_index == 0:
                    break
                self._seqword2rows[(word_index, seq_position)][0].append(target_index)
                self._seqword2rows[(word_index, seq_position)][1].append(row_index)
        
        self._seqword2rows = {
            (word_index, seq_position): (np.array(target_indices, dtype=np.int32), np.array(row_indices, dtype=np.int32))
            for (word_index, seq_position), (target_indices, row_indices) in self._seqword2rows.items()
        }
        

    #turns color_term string into reverse list of indices
    def color_term_to_indices(self, color_term):
        color_term = list(reversed(color_term.split()))
        color_term_indices = np.array([self.comp_xkcd.color_vocab.lookup_token(color_word) for color_word in color_term])
        return color_term_indices

    #turns color_term string into reverse list of indices, including <START> and <STOP> indices 
    def full_color_term_to_indices(self, color_term):
        color_term_indices = self.color_term_to_indices(color_term)
        color_term_indices = np.insert(color_term_indices, 0, 1)
        color_term_indices = np.append(color_term_indices, 2)
        return color_term_indices

    #color_term_indices is a list of indices that begins with 1 (the START token) and ends with 2 (the STOP token)
    def apply_model_to_colorspace(self, color_term_indices):
        color_term_indices = np.resize(color_term_indices, self.model.max_seq_len).reshape(1, self.model.max_seq_len)
        color_term_indices = torch.from_numpy(color_term_indices)

        p_word = []
        phi = []
        to_numpy = lambda tensor: tensor.cpu().detach().numpy()
        
        batch_generator = self.csd.generate_batches(
            batch_size=256, 
            shuffle=False, 
            drop_last=False,
            device=self.device
        )

        for _, batch in enumerate(batch_generator):
            color_term_input = color_term_indices.expand(batch['x_colors'].size(dim=0), -1)
            model_output = self.model(batch['x_colors'], color_term_input)
            p_word.append(to_numpy(model_output['S0_probability']))
            phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))
            
        p_word = np.vstack(p_word)
        p_word = p_word.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, p_word.shape[-2], p_word.shape[-1]))
        phi = np.vstack(phi)
        phi = phi.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, phi.shape[-2], phi.shape[-1]))
        return p_word, phi

    def apply_model_to_subset(self, target_indices):

        to_numpy = lambda tensor: tensor.cpu().detach().numpy() #put this somewhere else later
        phi = []
        p_word = []

        subset = torch.utils.data.Subset(self.comp_xkcd, target_indices)
        batch_generator = generate_batches(
            subset,
            batch_size=256, 
            shuffle=False, 
            drop_last=False,
            device=self.device
        )

        for _, batch in enumerate(batch_generator):
            model_output = self.model(batch['x_color_value'], batch['y_color_name'])

            p_word.append(to_numpy(model_output['S0_probability']))
            phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))

        p_word = np.vstack(p_word)
        phi = np.vstack(phi)

        return p_word, phi

    '''
    Contour plots the probability/applicability over colorspace of certain words in color_term, as specified by seq_positions
    Probabilities/applicabilities for those chosen words in color_term are multiplied together, so that the value calculated is
        the probability/applicability of word index 1 for seq position 1 AND word index 2 for seq position 2, etc.

    color_term is a string color description
    seq_positions is a list of integers specifying which words in the color description to caluclate the probabilities/applicabilities for
        indices are reversed, so that 0 is the last word in the string, and 1,2,... are indices of the words going left
        index len(color_term.split()) is the STOP token
    target can be either 'p_word' for probability or 'phi' for applicability

    '''
    def contour_plot_full_color_term(self, color_term,  seq_positions=-1, target='p_word', levels=[0.1, 0.5], 
                     linestyles=['--', '-'], figsize=(15, 5), title_prefix='', 
                     dim_reduce_func=np.mean, num_to_scatter=-1, scatter_seed=0):
        
        word_indices = self.full_color_term_to_indices(color_term)
        print("Translated to indices: " + str(word_indices))
        
        #figure setup and title creation
        #[START red bright burning STOP]
            #'Applicability (phi) across colorspace of 'red' acting as head and 'bright','burning' acting as modifiers in 'burning bright red'
            #'Applicability (phi) across colorspace of 'bright','burning' acting as modifiers in 'burning bright red'
            #'Applicability (phi) across colorspace of 'burning bright red' being a full description
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        if len(title_prefix) > 0:
            title_prefix = title_prefix.strip() + ' '       
        title = 'Applicability (phi) across colorspace of' if target == "phi" else 'Probability across colorspace of'       
        if seq_positions == -1 or len(seq_positions)+1==len(word_indices):
            title+=f' {color_term} as a full description'
        else:
            head = False
            modifiers = False
            for i in range(len(word_indices)-1):
                word = self.comp_xkcd.color_vocab._idx_to_token[word_indices[i+1]]
                if i in seq_positions:
                    if i == 0:
                        title+=f' {word} acting as head' 
                        head = True
                    elif head and not modifiers:
                        title+=f' and {word}'
                        modifiers = True
                    else:
                        title+=f' {word}'
                        modifiers = True
            if modifiers:
                title+=f' acting as modifier'
            if (np.array(seq_positions)>0).sum() > 1:
                title+='s'
            title+=f' in {color_term}'

        #default is to get activations for all words
        if seq_positions == -1:
            seq_positions = np.arange(len(word_indices)-1)
            print("Seq positions: " +  str(seq_positions))

        p_word, phi = self.apply_model_to_colorspace(word_indices)
        activations = p_word if target=='p_word' else phi
        
        xkcd_term_index = self.xkcd.color_vocab.lookup_token(color_term)
        comp_xkcd_word_indices = np.take(np.delete(word_indices, 0), seq_positions)
        print("Selected indices: " +  str(comp_xkcd_word_indices))
        
        if num_to_scatter > 0:
            random_state = np.random.RandomState(seed=scatter_seed)
            xkcd_row_indices = self._label2rows[xkcd_term_index]
            subset = random_state.choice(
                xkcd_row_indices, 
                size=min(num_to_scatter, len(xkcd_row_indices)), 
                replace=False
            )
            full_scatter_points = self.xkcd._original_data_matrix[subset]
        
        #About contourf
            # first two parameters are the grid over which contour is plotted
            # third parameter is the 3rd axis that is represented with a gradient
        #dim_reduce_func averages together points along the value/saturation dimension because each plot is 2D
        #[:,:,seq_positions, comp_xkcd_word_indices] chooses the probabilities/phis of word indices (given previous word indices) at correspding seq_positions,
            #.prod(axis=2) multiplies these values within the sequence
            #the resulting value for each sequence is the probability of choosing word index 1 for seq position 1 AND word index 2 for seq position 2, etc.
        
        im = axes[0].contourf(self.csd.h, self.csd.s, dim_reduce_func(activations, axis=2)[:,:,seq_positions, comp_xkcd_word_indices].prod(axis = 2).T, alpha=0.3)
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Hue")
        axes[0].set_ylabel("Saturation")
        if num_to_scatter > 0:
            axes[0].scatter(full_scatter_points[:, 0], full_scatter_points[:, 1], marker='x', color='black', alpha=0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        im = axes[1].contourf(self.csd.h, self.csd.v, dim_reduce_func(activations, axis=1)[:,:,seq_positions, comp_xkcd_word_indices].prod(axis = 2).T, alpha=0.3)
        plt.colorbar(im, ax=axes[1])
        axes[1].set_xlabel("Hue")
        axes[1].set_ylabel("Value")
        if num_to_scatter > 0:
            axes[1].scatter(full_scatter_points[:, 0], full_scatter_points[:, 2], marker='x', color='black', alpha=0.5)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title)

    #color_term is a color description string to be turned into a reversed list of indices
    #seq_positions should correspond to the the reversed order of the words in color_term
    #the points plotted are then those points where wordindx1 from reversed list is in seq_position1 AND wordindx2 in seq_position2 etc.
    def scatter_plot_subset_color_term(self, color_term, seq_positions, target='p_word', dim = '3d', levels=[0.1, 0.5], 
                                        linestyles=['--', '-'], figsize=(15, 5), title_prefix='', 
                                        num_to_scatter=-1, scatter_seed=0):
        
        #to stay consistent with contour_plot above, input position 0 is after START, so the input seq_positions is moved forward one spot
        seq_positions = np.array(seq_positions)+1

        color_term_indices = self.color_term_to_indices(color_term)
        
        #use intersection to get points where all words in color_term occur in their seq_positions
        if len(color_term_indices)==1:
            target_indices, row_indices = self._seqword2rows[(color_term_indices[0], seq_positions[0])]
        else:
            target_indices = reduce(np.intersect1d, [self._seqword2rows[(word_index, seq_position)][0] for word_index, seq_position in zip(color_term_indices, seq_positions)])
            row_indices = reduce(np.intersect1d, [self._seqword2rows[(word_index, seq_position)][1] for word_index, seq_position in zip(color_term_indices, seq_positions)])
        if len(target_indices) == 0:
            print("No points were found")
            return
        
        if num_to_scatter>-1:
            common_indices = np.arange(len(target_indices))
            random_state = np.random.RandomState(seed=scatter_seed)
            subset = random_state.choice(
                common_indices, 
                size=min(num_to_scatter, len(common_indices)), 
                replace=False
            )
            target_indices = target_indices[subset]
            row_indices = row_indices[subset]

        hsv_values = self.comp_xkcd._original_data_matrix[row_indices]
        p_word, phi = self.apply_model_to_subset(target_indices)
        activations = p_word if target=='p_word' else phi
        activations = activations[:,seq_positions, color_term_indices].prod(axis = 1)

        #pick out points whose full description is color_term to mark with triangle
        xkcd_row_indices = self._label2rows[self.xkcd.color_vocab.lookup_token(color_term)]
        common_indices = np.intersect1d(row_indices,xkcd_row_indices,return_indices=True)[1] #get indices in row_indices that are full descriptions
        xkcd_hsv_values = hsv_values[common_indices]
        xkcd_activations = activations[common_indices]

        if len(title_prefix) > 0:
            title_prefix = title_prefix.strip() + ' '

        if dim=='2d':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            img = axes[0].scatter(hsv_values[:,0],hsv_values[:,1], c = activations, marker = 'o', cmap = plt.hot(), alpha = 0.4)
            img_xkcd = axes[0].scatter(xkcd_hsv_values[:,0],xkcd_hsv_values[:,1], c = xkcd_activations, marker = '^', cmap = plt.hot(), s=65, alpha = 1)
            plt.colorbar(img, ax = axes[0])
            axes[0].set_xlabel('Hue')
            axes[0].set_ylabel('Saturation')
            axes[0].set_xlim(0,1)
            axes[0].set_ylim(0,1)
            img = axes[1].scatter(hsv_values[:,0],hsv_values[:,2], c = activations, marker = 'o', cmap = plt.hot(), alpha = 0.4)
            img_xkcd = axes[1].scatter(xkcd_hsv_values[:,0],xkcd_hsv_values[:,2], c = xkcd_activations, marker = '^', cmap = plt.hot(), s=65, alpha = 1)
            plt.colorbar(img, ax = axes[1])
            axes[1].set_xlabel('Hue')
            axes[1].set_ylabel('Value')
            axes[1].set_xlim(0,1)
            axes[1].set_ylim(0,1)
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            img = ax.scatter(hsv_values[:,0],hsv_values[:,1],hsv_values[:,2], c = activations, marker = 'o', cmap = plt.hot(), alpha = 0.4)
            img_xkcd = ax.scatter(xkcd_hsv_values[:,0], xkcd_hsv_values[:,1], xkcd_hsv_values[:,2], c = xkcd_activations, marker = '^', cmap = plt.hot(), s=65, alpha = 1)
            fig.colorbar(img)
            ax.set_xlabel('Hue')
            ax.set_ylabel('Saturation')
            ax.set_zlabel('Value')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_zlim(0,1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if target == "phi":
            plt.suptitle(f'Applicability (phi) of points in colorspace containing {color_term} at positions {seq_positions-1} from end')
        elif target == "p_word":
            plt.suptitle(f'Probability of points in colorspace containing {color_term} at positions {seq_positions-1} from end')
