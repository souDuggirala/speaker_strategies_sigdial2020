from magis_sigdial2020.datasets.xkcd.vectorized import XKCD, CompositionalXKCD
from magis_sigdial2020.utils.data import Dataset
import numpy as np



class TeacherGuidedXKCD(Dataset):
    def __init__(self, teacher_phi_path, xkcd_coordinate_system='x-y', compositional = False, max_seq_len = 6):
        self.compositional = compositional
        self.max_seq_len = max_seq_len
        if self.compositional:
            self.xkcd = CompositionalXKCD.from_settings(coordinate_system=xkcd_coordinate_system)
        else:
            self.xkcd = XKCD.from_settings(coordinate_system=xkcd_coordinate_system)
        #in the compositional case need to account for the padding "0" not in the dictionary
        self.vocab_size = len(self.xkcd.color_vocab)+1 if self.compositional else len(self.xkcd.color_vocab)
        self.teacher_phi = np.load(teacher_phi_path).astype(np.float32)
        self.split = None
        self.set_split("train")
        self._teacher_phi_path = teacher_phi_path

    def get_teacher_phi_path(self):
        return self._teacher_phi_path
    
    def set_split(self, split):
        self.xkcd.set_split(split)
        self.split = split

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

    def __len__(self):
        return len(self.xkcd)