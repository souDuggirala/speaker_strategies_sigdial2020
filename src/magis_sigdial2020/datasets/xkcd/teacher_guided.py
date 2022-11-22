from magis_sigdial2020.datasets.xkcd.vectorized import XKCD, CompositionalXKCD
from magis_sigdial2020.utils.data import Dataset
import numpy as np



class TeacherGuidedXKCD(Dataset):
    def __init__(self, teacher_phi_path, teacher_prob_path, xkcd_coordinate_system='x-y', compositional = False, max_seq_len = 6):
        
        self.teacher_phi = np.load(teacher_phi_path).astype(np.float32) if teacher_phi_path != -1 else None
        self.teacher_prob = np.load(teacher_prob_path).astype(np.float32) if teacher_prob_path != -1 else None
        #allowing either teacher_phi or teacher_prob to be None, but both cannot be None
        assert not (self.teacher_phi is None and self.teacher_prob is None)
        self._teacher_phi_path = teacher_phi_path
        self._teacher_prob_path = teacher_prob_path
        
        self.compositional = compositional
        self.max_seq_len = max_seq_len
        if self.compositional:
            self.xkcd = CompositionalXKCD.from_settings(coordinate_system=xkcd_coordinate_system)
        else:
            self.xkcd = XKCD.from_settings(coordinate_system=xkcd_coordinate_system)
        #in the compositional case need to account for the padding "0" not in the dictionary
        self.vocab_size = len(self.xkcd.color_vocab)+1 if self.compositional else len(self.xkcd.color_vocab)
        self.split = None
        self.set_split("train")

    def get_teacher_phi_path(self):
        return self._teacher_phi_path

    def get_teacher_prob_path(self):
        return self._teacher_prob_path
    
    def set_split(self, split):
        self.xkcd.set_split(split)
        self.split = split

    def __getitem__(self, index):
        output = self.xkcd[index]
        if self.split == "train":
            teacher_phi = self.teacher_phi[index] if self.teacher_phi is not None else -1
            teacher_prob = self.teacher_prob[index] if self.teacher_prob is not None else -1
        else:
            #seems to be that during validation, you want to compare against an "ideal" where full probability is put on the true answer
            #does output include START or no?....it shouldn't because its being directly compared to model output
            if self.compositional:
                seq_indices = np.arange(self.max_seq_len)
                teacher_phi = np.zeros((self.max_seq_len,self.vocab_size)).astype(np.float32)
                teacher_phi[seq_indices,output['y_color_name']] = 1
                #teacher prob does not have an axis for vocab_size, because otherwise there would be too many combinations of tokens to calculate the cumulative probability for
                    #the "ideal" in this case would be all 1's up until STOP?
                #I'm going to go with doing the same thing for both train and validation, because the full probability thing above doesn't make sense for this...
                teacher_prob = self.teacher_prob[index] if self.teacher_prob is not None else None
            else:
                teacher_phi = np.zeros(self.vocab_size).astype(np.float32)
                teacher_phi[output['y_color_name']] = 1
                #below (teacher prob and not compositional) is not a case I have thought much about so there may be an error here
                    #I only included it for the sake of completeness and not checking too many conditions
                    #for instance, cumulative probability is not a thing with non compositionality, so it would just be using probability as a loss instead of phi in the same exact way
                teacher_prob = np.zeros(self.vocab_size).astype(np.float32)
                teacher_prob[output['y_color_name']] = 1

        output['teacher_phi'] = teacher_phi
        output['teacher_prob'] = teacher_prob
        return output

    def __len__(self):
        return len(self.xkcd)