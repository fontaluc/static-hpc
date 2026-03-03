import torch
from torch import nn
from pcn import utils
import pcn
import numpy as np

class Layer(nn.Module):
    def __init__(
        self, in_size, out_size, act_fn, c, glorot_init=False, device=None
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.c = c

        self.K = int(self.c * self.in_size)
        cols = [torch.randperm(self.in_size)[:self.K].unsqueeze(1) for _ in range(self.out_size)]
        self.row_idx = torch.cat(cols, dim=1)
        self.col_idx = torch.arange(self.out_size).repeat(self.K, 1)
        self.weights = nn.Parameter(torch.empty((self.in_size, self.out_size)), device=device)
        self.bias = nn.Parameter(torch.empty((self.out_size)), device=device)
        
        self._reset_grad()

        if glorot_init:
            self._reset_params_glorot() 
        else:
            self._reset_params()

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_params(self):
        self.weights[self.row_idx, self.col_idx] = nn.init.normal_(torch.empty((self.K, self.out_size)), mean=0, std=2/np.sqrt(self.in_size))

    def _reset_params_glorot(self):
        self.weights[self.row_idx, self.col_idx] = nn.init.xavier_uniform_(torch.empty((self.K, self.out_size)))

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
        delta=False,
        glorot_init=False,
        device=None
    ):
        super().__init__(
            in_size, 
            out_size, 
            act_fn, 
            c,
            glorot_init,
            device
        )
        self.in_mean = pcn.utils.set_tensor(in_mean)
        self.f = f
        self.k = int(self.f*self.out_size)
        self.use_bias = use_bias
        self.delta = delta

    def forward(self, inp):
        h = torch.matmul(inp, self.weights) + self.bias
        # kWTA: keep only top-k activations per row
        if self.f < 1:
            topk_vals, _ = torch.topk(h, self.k + 1, dim=1)
            kth_vals = topk_vals[:, -1].unsqueeze(1)
            h = h - kth_vals        
        if self.act_fn == torch.heaviside:
            out = self.act_fn(h, pcn.utils.set_tensor(torch.zeros(1)))
        else:
            out = self.act_fn(h)
        return out
    
    def update_weights(self, inp, target, out):
        y = target
        if self.delta:
            y = y - out
        self.grad["weights"] = torch.matmul((inp - self.in_mean).T, y)
        if self.use_bias:
            self.grad["bias"] = torch.sum(y, axis=0)