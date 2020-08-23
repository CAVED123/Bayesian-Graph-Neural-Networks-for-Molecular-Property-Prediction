import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #print(2 * torch.log(sig_p / sig_q).sum())
    #print(((sig_q / sig_p).pow(2)).sum())
    #print((((mu_p - mu_q) / sig_p).pow(2)).sum())
    return KLD


class BayesLinear(nn.Module):
    
    """
    Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
    the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
    with gaussian priors.
    """
    
    def __init__(self, n_in, n_out, prior_sig, bias = True):
        
        super(BayesLinear, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.prior_sig = prior_sig
        
        # initialise mu for weights
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.02, 0.02))
        
        # initialise mu for biases
        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.02, 0.02))

    
    def init_rho(self, p_min, p_max):
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(p_min, p_max))
        if self.bias:
            self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(p_min, p_max))     


    def forward(self, X, sample=False):
        
        if sample:

            ### weights
            
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20) # compute stds for weights
            act_W_mu = torch.mm(X, self.W_mu)  # activation means
            act_W_std = torch.sqrt(torch.clamp_min(torch.mm(X.pow(2), std_w.pow(2)),1e-6)) # actiavtion stds
            
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
            act_W_out = act_W_mu + act_W_std * eps_W # sample weights from 'posterior'
            
            output = act_W_out
            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w)
            
            if self.bias:
                
                std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20) # compute stds for biases
                eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
                act_b_out = self.b_mu + std_b * eps_b # sample biases from 'posterior'
                output += act_b_out.unsqueeze(0).expand(X.shape[0], -1)
                kld += KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.b_mu, sig_q=std_b)
       
            return output, kld
        
        
        else:
            
            output = torch.mm(X, self.W_mu)
            
            # kld is just standard regularisation term
            kld = (0.5*((self.W_mu / self.prior_sig).pow(2))+torch.log(self.prior_sig*torch.sqrt(torch.tensor(2*np.pi)))).sum()
            
            if self.bias:
                output += self.b_mu.expand(X.size()[0], self.n_out)
                
                kld += (0.5*((self.b_mu / self.prior_sig).pow(2))+torch.log(self.prior_sig*torch.sqrt(torch.tensor(2*np.pi)))).sum()
            
            
            return output, kld
        
        
        

    