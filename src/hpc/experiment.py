import argparse
import os
from filelock import FileLock
import pandas as pd
import torch
from hpc.layers import SparseLayer
import pcn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from hpc import utils

def main(cf):
    
    cifar10_train = torch.load(f'data/cifar10_train.pt')
    ec_train = cifar10_train['ec']
    ec_train = pcn.utils.set_tensor(ec_train)

    cifar10_valid = torch.load(f'data/cifar10_valid.pt')
    ec_valid = cifar10_valid['ec']
    ec_valid = pcn.utils.set_tensor(ec_valid)

    pcn.utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)    

    EC_DG = SparseLayer(
        in_size=cf.n_ec, 
        out_size=cf.n_dg, 
        act_fn=cf.act_fn,
        c=1,
        f=cf.f_dg
    )

    DG_CA3 = SparseLayer(
        in_size=cf.n_dg,
        out_size=cf.n_ca3, 
        act_fn=cf.act_fn,
        c=1,
        f=cf.f_ca3
    )
    
    with torch.no_grad():
        dg_train = EC_DG(ec_train)
        ca3_train = DG_CA3(dg_train)
        dg_valid = EC_DG(ec_valid)
        ca3_valid = DG_CA3(dg_valid)

    # Average correlation
    ca3 = ca3_train[:cf.N_patterns]
    indices = torch.triu_indices(cf.N_patterns, cf.N_patterns, offset=1)
    R = torch.corrcoef(ca3).abs()
    avg_corr = R[indices[0], indices[1]].mean().item()

    # Validation error
    
    train_tensordataset = TensorDataset(ca3_train, ec_train)
    train_loader = DataLoader(
        train_tensordataset, 
        cf.batch_size, 
        shuffle=True, 
        worker_init_fn=pcn.utils.seed_worker, 
        generator=g
    )
    valid_tensordataset = TensorDataset(ca3_valid, ec_valid)
    valid_loader = DataLoader(
        valid_tensordataset, 
        cf.batch_size, 
        shuffle=True, 
        worker_init_fn=pcn.utils.seed_worker, 
        generator=g
    )     

    device = pcn.utils.DEVICE
    criterion = nn.MSELoss()
    model = nn.Sequential(
        nn.Linear(in_features=cf.n_ca3, out_features=cf.n_ca1),
        nn.ReLU(), 
        nn.Linear(in_features=cf.n_ca1, out_features=cf.n_ec),
        nn.Tanh()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr)
    
    errors_train, errors_valid = utils.train_model(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        criterion, 
        device,
        lambda_reg=cf.lambda_reg,  
        n_epochs=cf.n_epochs)
    
    valid_error = min(errors_valid)

    # Lock file to prevent overwriting when multiple processes run
    with FileLock(f"sparsity_experiments.csv.lock"):
        print('Lock acquired.')
        data = [cf.f_dg, cf.f_ca3, avg_corr, valid_error]
        if os.path.exists("outputs/sparsity_experiments.csv"):
            df = pd.read_csv("outputs/sparsity_experiments.csv")
            df.loc[len(df)] = data
        else:
            df = pd.DataFrame([data], columns=['DG sparsity', 'CA3 sparsity', 'Average correlation', 'Validation error'])
        df.to_csv('outputs/sparsity_experiments.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that evaluate the PC model according to different metrics to choose the right number of EC units"
    )
    parser.add_argument("--f_dg", type=float, default=0.01, help="Enter DG sparsity")
    parser.add_argument("--f_ca3", type=float, default=0.06, help="Enter CA3 sparsity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Enter learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Enter number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = pcn.utils.AttrDict()

    cf.seed = args.seed
    cf.n_ec = 300
    p = cf.n_ec/(1.1*10**5)
    cf.n_dg = round(1.2*10**6*p)
    cf.n_ca3 = round(2.5*10**5*p)
    cf.n_ca1 = round(3.9*10**5*p)

    cf.f_dg = args.f_dg
    cf.f_ca3 = args.f_ca3

    cf.act_fn = torch.relu
    cf.N_patterns = 640  
    cf.lr = args.lr
    cf.n_epochs = args.n_epochs
    cf.batch_size = 64
    cf.lambda_reg = 0    
        
    main(cf)

