import numpy as np
import torch
import torch.nn.functional as F


def compute_accuracy(y_pred, y_target, mask_padding = False):
    y_target = y_target.cpu()
    _, y_pred_indices = y_pred.cpu().max(dim=-1)
    if mask_padding:
        y_pred_indices = y_pred_indices.masked_fill(y_target==0, -1) #wherever y_target is 0, y_pred_indices is -1 so they're not counted as equal
        num_indices = (y_target!=0).sum().item()
    else:
        num_indices = torch.numel(y_target).item()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / num_indices * 100

def compute_perplexity(y_pred, y_true, apply_softmax=False):
    if apply_softmax:
        y_pred = F.softmax(y_pred, dim=1)
        
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    indices = np.arange(len(y_true))
    
    return 2**np.mean(-np.log2(y_pred[indices, y_true]))

def compute_perplexity_seq(y_pred, y_true, apply_softmax=False):
    if apply_softmax:
        y_pred = F.softmax(y_pred, dim=2)
        
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    batch_indices = np.arange(y_true.shape[0])
    seq_indices = np.arange(y_true.shape[1])

    return 2**np.mean(np.sum(-np.log2(y_pred[:, seq_indices, y_true][batch_indices, batch_indices]), axis = 1))