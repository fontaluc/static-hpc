import torch
from torch import nn
from pcn import utils
import pcn
import numpy as np

class Layer(nn.Module):
    def __init__(
        self, in_size, out_size, act_fn, c, glorot_init=False, device=utils.DEVICE
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.c = c
        self.device = device

        self.K = int(self.c * self.in_size)
        cols = [torch.randperm(self.in_size)[:self.K].unsqueeze(1) for _ in range(self.out_size)]
        self.row_idx = torch.cat(cols, dim=1)
        self.col_idx = torch.arange(self.out_size).repeat(self.K, 1)
        self.weights = torch.empty((self.in_size, self.out_size), device=self.device)
        # always create a bias tensor but initialize to zero unless user chooses otherwise
        self.bias = torch.zeros((self.out_size), device=self.device)
        
        self._reset_grad()

        if glorot_init:
            self._reset_params_glorot() 
        else:
            self._reset_params()

        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_params(self):
        self.weights[self.row_idx, self.col_idx] = nn.init.normal_(torch.empty((self.K, self.out_size), device=self.device), mean=0, std=2/np.sqrt(self.in_size))

    def _reset_params_glorot(self):
        self.weights[self.row_idx, self.col_idx] = nn.init.xavier_uniform_(torch.empty((self.K, self.out_size), device=self.device))

class SparseLayer(Layer):
    def __init__(
        self, 
        in_size, 
        out_size, 
        act_fn, 
        c, 
        f,
        in_mean = torch.zeros(1),
        use_bias=False,
        glorot_init=False,
        device=utils.DEVICE
    ):
        super().__init__(
            in_size, 
            out_size, 
            act_fn, 
            c,
            glorot_init,
            device
        )
        self.in_mean = utils.set_tensor(in_mean, self.device)
        self.f = f
        self.k = int(self.f*self.out_size)
        self.use_bias = use_bias
        self.inp = None
        self.out = None

    def apply_inhibition(self, h):
        if self.act_fn == torch.heaviside or self.act_fn == torch.relu:
            topk_vals, _ = torch.topk(h, self.k + 1, dim=1)
            kth_vals = topk_vals[:, -1].unsqueeze(1)
            h = h - kth_vals

            if self.act_fn == torch.heaviside:
                z = self.act_fn(h, utils.set_tensor(torch.zeros(1), self.device))
            else:
                z = self.act_fn(h)
        else:
            z = self.act_fn(h)
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            z = torch.zeros_like(h)
            z.scatter_(1, topk_idx, topk_vals)
        return z


    def forward(self, inp):
        self.inp = inp.clone()
        # compute linear response; optionally add bias
        h = torch.matmul(inp, self.weights)
        if self.use_bias:
            h = h + self.bias
        # kWTA: keep only top-k activations per row
        if self.f < 1:
            self.out = self.apply_inhibition(h)
        else:
            self.out = self.act_fn(h)

        return self.out
    
    def update_weights(self, target, pred=None):
        y = target
        if pred is not None:
            y = y - pred
        self.grad["weights"] = torch.matmul((self.inp - self.in_mean).T, y)
        if self.use_bias:
            self.grad["bias"] = torch.sum(y, axis=0)