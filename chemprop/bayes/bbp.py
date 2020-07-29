import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out




class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term)




class BayesLinear_Normalq(nn.Module):
    
    """
    Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters.
    The likelihood of the weight samples under the prior and the approximate posterior are returned with each forward pass.
    """
    
    def __init__(self, n_in, n_out, prior_class, bias = True):
        
        super(BayesLinear_Normalq, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.prior = prior_class

        # initialise mu and p for weights
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.05, 0.05))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-2, -1))
        
        # initialise mu and p for biases
        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.05, 0.05))
            self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-2, -1))
        

    def forward(self, X, sample=True):

        if not sample:
            
            output = torch.mm(X, self.W_mu)
            
            if self.bias:
                output += self.b_mu.expand(X.size()[0], self.n_out)
            
            return output, 0, 0

        else:

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            W = self.W_mu + 1 * std_w * eps_W
            
            output = torch.mm(X, W)
            
            lqw = isotropic_gauss_loglike(W, self.W_mu, std_w)
            lpw = self.prior.loglike(W)
            
            if self.bias:
                eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())
                std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)
                b = self.b_mu + 1 * std_b * eps_b
                
                output += b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
                
                lqw += isotropic_gauss_loglike(b, self.b_mu, std_b)
                lpw += self.prior.loglike(b)
                
                
            return output, lqw, lpw













