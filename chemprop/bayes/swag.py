"""
    implementation of SWAG
    based on the original repo: https://github.com/wjmaddox/swa_gaussian
"""

import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal
import copy

from chemprop.bayes_utils import flatten, unflatten_like



def swag_parameters(module, params, no_cov_mat=True):
    """
    module: submodule of base model
    params: list of params
    no_cov_mat: True means SWAG DIAG
    """
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        
        # registers buffers for first moment and second moment
        # initialised at zero?
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())
        
        # if full SWAG, registers D (sqrt of covariance matrix)
        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        # append to list of SWAG parameters
        # HAVE INSERTED A HACK HERE TO COPE WITH CACHED_ZERO_VECTOR
        if name == 'cached_zero_vector':
            params.insert(0,(module, name))    
        else:
            params.append((module, name))
        



class SWAG(torch.nn.Module):
    """
    SWAG module which builds upon a base NN module
    """
    
    def __init__(self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30):
        super(SWAG, self).__init__()

        # adds buffer to module, counting number of models
        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        
        # empty list
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        # deepcopy of base model
        self.base = copy.deepcopy(base)
        
        # applies swag_parameters function to each submodule of base (loop over submodules is implicit)
        self.base.apply(
            lambda module: swag_parameters(
                module=module, params=self.params, no_cov_mat=self.no_cov_mat
            )
        )
        
    
    ### NOT SURE WHAT THIS IS FOR? DO CLASSES HAVE TO HAVE A FORWARD?
    def forward(self, *input):
        return self.base(*input)



    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)



    def sample_blockwise(self, scale, cov, fullrank):
        """
        samples blockwise: assumes at most covariance within each module
        """
        
        # self.params is a list of submodules and names
        for module, name in self.params:
            
            # first moment
            mean = module.__getattr__("%s_mean" % name)
            
            # second moment
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            
            # random numbers from normal distribution with mean zero and variance 1            
            eps = torch.randn_like(mean)

            # variance for each parameter
            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            # produce a sample from the diagonal variance (scaled)
            scaled_diag_sample = scale * torch.sqrt(var) * eps

            # if we have covariance...
            if cov is True:
                
                # get our D matrix
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                
                # new tensor: number of weights X 1, normally distributed
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                
                # cov sample
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, w)




    def sample_fullrank(self, scale, cov, fullrank):
        """
        samples fullrank: assumes covariance between submodules
        """
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample)
                               # fix later




    def collect_model(self, base_model):
        """
        updates first moment, second moment and sqrt of cov matrix
        """
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)






