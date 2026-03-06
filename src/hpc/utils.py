from sklearn.decomposition import PCA
import torch
from hpc.models import PatternAssociator, PATrainer
import pcn
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
from typing import *
from IPython.display import Image, display, clear_output
import os
import seaborn as sns

def best_lr(lrs, train_loader, test_loader, in_size, out_size, in_mean, act_fn, use_bias, delta, f, c=1, num_patterns=20):
    mean_errors = []
    min_error = float('inf')
    with torch.no_grad():
        for lr in lrs:
            model = PatternAssociator(
                in_size=in_size, 
                in_mean=in_mean,
                out_size=out_size, 
                act_fn=act_fn, 
                use_bias=use_bias, 
                c=c, 
                f=f, 
                delta=delta,
                glorot_init=True
            )
            optimizer = pcn.optim.get_optim(
                        [model.layer],
                        "SGD",
                        lr,
                        batch_scale=True,
                        grad_clip=None,
                        weight_decay=None
                    )
            trainer = PATrainer(model, optimizer, nn.MSELoss())
            trainer.train(train_loader, 0)
            errors = trainer.test(test_loader)
            mean_error = np.mean(errors)
            if mean_error < min_error:
                min_error = mean_error
                argmin = lr
            mean_errors.append(mean_error)
    return (argmin, min_error), mean_errors

def get_representations(data_loader, model, n_max_iters, step_tolerance, init_std, fixed_preds_test):
    img = []
    vc = []
    ec = []
    labels = []
    with torch.no_grad():
        for img_batch, label_batch in tqdm(data_loader):
            img += img_batch.tolist()
            labels += label_batch.tolist()
            # Get EC activities for episodes
            img_batch = pcn.utils.set_tensor(img_batch)
            model.test_batch(
                img_batch, 
                n_iters=n_max_iters, 
                step_tolerance=step_tolerance, 
                init_std=init_std, 
                fixed_preds=fixed_preds_test
            )
            vc += model.mus[1].to('cpu').tolist()
            ec += model.mus[0].to('cpu').tolist()        

    ec = torch.tensor(ec)
    vc = torch.tensor(vc)
    img = torch.tensor(img)
    labels = torch.tensor(labels)

    return img, vc, ec, labels

def get_supports(x):   
    support = (x > 0)
    supports = [tuple(row.nonzero(as_tuple=True)[0]) for row in support]    
    return set(supports)

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, regularization_type='L2', lambda_reg=1e-6, n_epochs=50):
    errors_train = []
    errors_valid = []

    # Set the model to training mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.train()

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        epoch_loss = 0

        # For each batch of data
        for x_batch, y_batch in train_loader:
            # Forward pass
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)

            # Compute loss value
            y_batch = y_batch.to(device)
            loss = criterion(y_pred, y_batch)

            # Apply L1 regularization
            if regularization_type == 'L1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += lambda_reg * l1_norm
            
            # Apply L2 regularization
            elif regularization_type == 'L2':
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss += lambda_reg * l2_norm

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Accumulate data for epoch metrics: loss
                epoch_loss += loss.item()

        # Compute epoch metrics
        mean_loss = epoch_loss / len(train_loader)
        errors_train.append(mean_loss)

        if (epoch + 1) % 5 == 0:

            with torch.no_grad():
                total_loss_valid = 0
                for x_valid, y_valid in valid_loader:
                    x_valid = x_valid.to(device)
                    y_valid_pred = model(x_valid)

                    # Compute loss value
                    y_valid = y_valid.to(device)
                    loss_valid = criterion(y_valid_pred, y_valid).item()
                
                    total_loss_valid += loss_valid
                mean_loss_valid = total_loss_valid / len(valid_loader)

            errors_valid.append(mean_loss_valid)
            
            print(
                f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Training loss: {mean_loss:.5f}. Validation loss: {mean_loss_valid:.5f}."
            )
    return errors_train, errors_valid

def dimensionality(X):
    # Use PCA to determine the number of dimensions needed to explain 90% of the variance in the data
    pca = PCA()
    pca.fit(X)
    cumsum = pca.explained_variance_ratio_.cumsum()
    return min(np.where(cumsum > 0.9)[0]).item() + 1

def plot_autoencoder_stats(
        x: Tensor = None,
        x_hat: Tensor = None,
        z: Tensor = None,
        y: Tensor = None,
        epoch: int = None,
        train_loss: List = None,
        valid_loss: List = None,
        classes: List = None,
        dimensionality_reduction_op: Optional[Callable] = None,
        size: Tuple = None
) -> None:
    """
    An utility 
    """
    # -- Plotting --
    f, axarr = plt.subplots(2, 2, figsize=(20, 20))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    ax.plot(np.arange(epoch + 1), train_loss, color="black")
    ax.plot(np.arange(epoch + 1), valid_loss, color="gray", linestyle="--")
    ax.legend(['Training error', 'Validation error'])

    # Latent space
    ax = axarr[0, 1]

    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # If you want to use a dimensionality reduction method you can use
    # for example TSNE by projecting on two principal dimensions
    # TSNE.fit_transform(z)
    if dimensionality_reduction_op is not None:
        z = dimensionality_reduction_op(z)

    palette = sns.color_palette()
    for c in classes:
        ax.scatter(*z[y.numpy() == c].T, c=palette[int(c)], marker='o')

    ax.legend(classes)

    # Inputs
    ax = axarr[1, 0]
    ax.set_title('Inputs')
    pcn.plotting.plot_samples(ax, x, size=size)

    # Reconstructions
    ax = axarr[1, 1]
    ax.set_title('Reconstructions')
    pcn.plotting.plot_samples(ax, x_hat, size=size)

    tmp_img = "tmp_ae_out.png"
    plt.savefig(tmp_img)
    plt.close(f)
    clear_output(wait=True)
    display(Image(filename=tmp_img))
    

    os.remove(tmp_img)