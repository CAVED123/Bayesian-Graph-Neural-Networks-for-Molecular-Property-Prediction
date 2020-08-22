
import torch
import torch.nn as nn
import numpy as np
            
            
def enable_dropout(m):
    """
    Parameters
    ----------
    m : submodule of a Pytorch model

    Activates dropout layers independently

    """
    if type(m) == nn.Dropout:
        m.train()
        

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

class scheduler_const():
    """
    mock scheduler for constant learning rates
    TAKES IN LIST
    """
    def __init__(self, lr_list):
        self.lr_list = lr_list
    def get_last_lr(self):
        return self.lr_list


def neg_log_like(output, target, sigma):
    """
    Negative gaussian log likelihood (scaled, per example)
    """
    
    exponent = -0.5*torch.sum(
        (target - output)**2/sigma**2
        , 1)

    log_coeff = -torch.sum(torch.log(sigma)) - len(sigma) * torch.log(torch.sqrt(torch.tensor(2*np.pi)))
    
    scale = 1 / len(exponent)
    
    return - scale * (log_coeff + exponent).sum()