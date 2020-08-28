
import numpy as np
import torch
import torch.nn as nn
import gpytorch

from typing import List

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler


class GPLayer(gpytorch.models.ApproximateGP):
    """
    Variational GP layer to take inputs from MPNN featuriser
    Generates num_dim outputs
    """
    
    def __init__(self, inducing_points, num_dim):
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points = inducing_points.size(-2),
            batch_shape = torch.Size([num_dim])
        )

        
        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, 
                inducing_points, 
                variational_distribution, 
                learn_inducing_locations=True
            ), num_tasks=num_dim
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_dim])),
            batch_shape=torch.Size([num_dim])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



    
    
class DKLMoleculeModel(gpytorch.Module):
    """
    Deep Kernel Learning model (MPNN featuriser + variational GP layer)
    """
    
    def __init__(self, feature_extractor, gp_layer):
        
        super(DKLMoleculeModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer

    def forward(self, *input):
        
        features = self.feature_extractor(*input)
        res = self.gp_layer(features)
        
        return res
    



def initial_inducing_points(
        train_data_loader,
        feature_extractor,
        args
        ):
    """
    Initialises inducing points
    Shape: num_tasks X num_inducing_points X hidden_size
    """
    inducing_points = []
    for batch in train_data_loader:
        mol_batch = batch.batch_graph()
        inducing_points.extend(feature_extractor(mol_batch))
    inducing_points = torch.stack(inducing_points)[:args.num_inducing_points]
    inducing_points = inducing_points.repeat(args.num_tasks,1,1)
    
    return inducing_points
    
    
    





def predict_std_gp(
    model: nn.Module, 
    data_loader: MoleculeDataLoader,
    args: TrainArgs,
    scaler: StandardScaler,
    likelihood) -> List[List[float]]:
    
    model.eval()
    preds = []

    for batch in data_loader:
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        with torch.no_grad():
            # get variance
            batch_preds = likelihood(model(mol_batch, features_batch)).variance

        # take sqr root
        batch_preds = np.sqrt(batch_preds.data.cpu().numpy())

        # Inverse scale
        if scaler is not None:
            batch_preds = batch_preds * np.array(scaler.stds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    