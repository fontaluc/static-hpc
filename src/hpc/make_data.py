from hpc import utils
import pcn
import torch
from pcn import datasets
from pcn.models import PCModel
import argparse
import numpy as np

def main(cf):

    pcn.utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    train_dataset, valid_dataset, test_dataset, size = pcn.utils.get_datasets(
        cf.dataset, 
        cf.train_size, 
        cf.test_size, 
        cf.normalize, 
        g
    )
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size, pcn.utils.seed_worker, g)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size, pcn.utils.seed_worker, g)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size, pcn.utils.seed_worker, g)

    if cf.dataset == 'mnist':
        cf.n_vc = 450
    elif cf.dataset == 'fmnist':
        cf.n_vc = 750
    else:
        cf.n_vc = 2000

    nodes = [cf.n_ec, cf.n_vc, np.prod(size)]
        
    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}-n_ec={cf.n_ec}" 
    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"../unsupervised-pcn/models/{model_name}.pt", map_location=pcn.utils.DEVICE, weights_only=True))

    img_train, vc_train, ec_train, labels_train = utils.get_representations(
        train_loader, 
        model, 
        cf.n_train_iters, 
        cf.step_tolerance, 
        cf.init_std, 
        cf.fixed_preds_train
    )
    cifar10_train = {'img': img_train.cpu(), 'vc': vc_train.cpu(), 'ec': ec_train.cpu(), 'labels': labels_train}
    torch.save(cifar10_train, f'data/cifar10_train.pt')

    img_valid, vc_valid, ec_valid, labels_valid = utils.get_representations(
        valid_loader, 
        model, 
        cf.n_train_iters, 
        cf.step_tolerance, 
        cf.init_std, 
        cf.fixed_preds_train
    )
    cifar10_valid = {'img': img_valid.cpu(), 'vc': vc_valid.cpu(), 'ec': ec_valid.cpu(), 'labels': labels_valid}
    torch.save(cifar10_valid, f'data/cifar10_valid.pt')

    img_test, vc_test, ec_test, labels_test = utils.get_representations(
        test_loader, 
        model, 
        cf.n_test_iters, 
        cf.step_tolerance, 
        cf.init_std, 
        cf.fixed_preds_test
    )
    cifar10_test = {'img': img_test.cpu(), 'vc': vc_test.cpu(), 'ec': ec_test.cpu(), 'labels': labels_test}
    torch.save(cifar10_test, f'data/cifar10_test.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that evaluate the PC model according to different metrics to choose the right number of EC units"
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='cifar10', help="Enter dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = pcn.utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = None
    cf.test_size = None
    cf.normalize = True
    cf.batch_size = 64

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.use_bias = True
    cf.n_ec = 300
    cf.act_fn = "tanh"
    cf.kaiming_init = False
    cf.positive = False

    main(cf)