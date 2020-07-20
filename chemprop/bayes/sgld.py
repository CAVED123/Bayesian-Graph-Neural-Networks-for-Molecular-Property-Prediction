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





class pSGLD(Optimizer):
    """
    RMSprop preconditioned SGLD using pytorch rmsprop implementation.
    """

    def __init__(self, params, args, lr=required, weight_decay=0, alpha=0.99, eps=1e-8, centered=False, addnoise=True):

        #weight_decay = 1 / (norm_sigma ** 2)
        
        self.args = args

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)


    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            
            # loop through parameters within a group
            for p in group['params']:
                
                # if no gradient, move to next parameter object
                if p.grad is None:
                    continue
                
                # d_p is the loss gradient
                d_p = p.grad.data
                
                # if parameter group is labelled addnoise=True
                if group['addnoise']:
                    
                    ##### preconditioning steps (1)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p.data)
                        if group['centered']:
                            state['grad_avg'] = torch.zeros_like(p.data)
                    square_avg = state['square_avg']
                    alpha = group['alpha']
                    state['step'] += 1
                    
                    # apply weight decay
                    if weight_decay != 0:
                        d_p.add_(p.data, alpha=weight_decay)
                    
                    ##### preconditioning steps (2)
                    # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                    square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)
                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                        avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                    else:
                        avg = square_avg.sqrt().add_(group['eps'])
                    
                    # normal noise for each p, take grad step
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1)
                    langevin_noise /= (np.sqrt(group['lr']) * np.sqrt(self.args.train_data_size))
                    p.data.add_(0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg), alpha=-group['lr'])
                    
                
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













    
    
    
    
    
    
    
