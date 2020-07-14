import torch
from .swag import SWAG
from chemprop.train.train import train



class scheduler_const():
    """
    mock scheduler for constant learning rate
    """
    def __init__(self, lr):
        self.lr = lr
    def get_lr(self):
        return [self.lr]



def train_swag(
        model,
        train_data_loader,
        loss_func,
        args):
    

    # define no_cov_mat from cov_mat
    if args.cov_mat:
        no_cov_mat = False
    else:
        no_cov_mat = True
    
    # instantiate SWAG model
    swag_model = SWAG(
        model,
        no_cov_mat,
        args.max_num_models,
        var_clamp=1e-30
    )

    # define optimiser
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_swag, momentum=args.momentum_swag, weight_decay=args.wd_swag)
    
    # define scheduler
    scheduler = scheduler_const(args.lr_swag)

    print("----------SWAG training----------")
    
    # training loop
    n_iter = 0
    for epoch in range(args.epochs_swag):
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                swag_model=swag_model
            )
        
    return swag_model

    
