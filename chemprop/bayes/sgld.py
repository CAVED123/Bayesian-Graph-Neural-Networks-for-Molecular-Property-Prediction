"""
    implementation of SGLD
    based on Javier Antoran's implementation of SGLD for homoscedastic regression
    source: https://github.com/JavierAntoran/Bayesian-Neural-Networks
"""

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch





class SGLD(Optimizer):
    """
    SGLD optimiser based on pytorch's SGD.
    Note that weight_decay = 1 / (norm_sigma ** 2) / self.args.train_data_size
    
    """

    def __init__(self, params, args, lr=required, weight_decay=0, addnoise=True):

        # set weight_decay from gaussian prior sigma
        #weight_decay = 1 / (norm_sigma ** 2)
        
        self.args = args

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # inherit from base class
        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)


    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        # loop through groups in param dict
        for group in self.param_groups:

            weight_decay = group['weight_decay']

            # loop through parameters within a group
            for p in group['params']:
        
                if p.grad is None:
                    continue # if grad is None, move to next param
                
                d_p = p.grad.data # grad
                
                # if parameter group is labelled addnoise=True
                if group['addnoise']:
                    
                    if weight_decay != 0:
                        #factor = weight_decay / self.args.train_data_size
                        d_p.add_(p.data, alpha=weight_decay)
                    
                    # normal noise for each p
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1)
                    langevin_noise /= (np.sqrt(group['lr']) * np.sqrt(self.args.train_data_size))
                    
                    # gradient step
                    p.data.add_(0.5 * d_p + langevin_noise, alpha=-group['lr'])
                
                # gradient step for log_noise parameter
                else:
                    p.data.add_(d_p, alpha=-group['lr'])

        return loss




def loss_sgld(output, target, sigma):
    """
    Computes scaled Gaussian log likelihood
    """
    
    exponent = -0.5*torch.sum(
        (target - output)**2/sigma**2
        , 1)

    log_coeff = -torch.sum(torch.log(sigma))
    
    scale = 1 / len(exponent)
    
    return - scale * (log_coeff + exponent).sum()













    
    
    
    
    
    
    
