import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable




def data_loss_bbp(output, target, sigma):
    """
    Gaussian log likelihood (scaled, per example)
    """
    
    exponent = -0.5*torch.sum(
        (target - output)**2/sigma**2
        , 1)

    log_coeff = -torch.sum(torch.log(sigma))
    
    scale = 1 / len(exponent)
    
    return - scale * (log_coeff + exponent).sum()



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
            assert np.all(np.isfinite(self.W_p.detach().numpy()))
            assert np.all(np.isfinite(X.detach().numpy()))
            
            
            act_W_mu = torch.mm(X, self.W_mu)  # activation means
            assert np.all(np.isfinite(act_W_mu.detach().numpy()))
            act_W_std = torch.sqrt(torch.clamp_min(torch.mm(X.pow(2), std_w.pow(2)),1e-6)) # actiavtion stds

            assert np.all(np.isfinite(act_W_std.detach().numpy()))
            #assert torch.all(act_W_std > 0)
            
            # try torch.randn_like(act_W_std)
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
            assert np.all(np.isfinite(eps_W.detach().numpy()))
            act_W_out = act_W_mu + act_W_std * eps_W # sample weights from 'posterior'
            
            output = act_W_out
            assert np.all(np.isfinite(output.detach().numpy()))
            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w)
            
            if self.bias:
                
                ### biases
                
                std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20) # compute stds for biases
                assert torch.all(std_b > 0)
                
                eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
                act_b_out = self.b_mu + std_b * eps_b # sample biases from 'posterior'
            
                output += act_b_out.unsqueeze(0).expand(X.shape[0], -1)
    
                kld += KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.b_mu, sig_q=std_b)
       
            
            assert np.all(np.isfinite(output.detach().numpy()))
            assert np.all(np.isfinite(kld.detach().numpy()))
            return output, kld
        
        
        else:
            
            output = torch.mm(X, self.W_mu)
            
            # kld is just standard regularisation term
            kld = 0.5*((self.W_mu / self.prior_sig).pow(2)).sum()
            
            if self.bias:
                output += self.b_mu.expand(X.size()[0], self.n_out)
                
                kld += 0.5*((self.b_mu / self.prior_sig).pow(2)).sum()
            
            
            return output, kld
        
        
        

    