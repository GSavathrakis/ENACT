import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, device):
        super().__init__()

        self.gaussian_kernel = (1./(sigma*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*sigma-1),3*sigma), 2)/(2*torch.pow(torch.Tensor([sigma]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])

        self.W_prob = nn.Linear(d_model, 1)

        self._reset_parameters()

        self.device = device

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, q_row, q_col, value):
        v = value.flatten(1,2)
        bs, spat, feats = q_row.shape
        prob_q = F.softmax(self.W_prob(q_row).squeeze(-1), -1) + 1e-8

        entropy = -prob_q*torch.log(prob_q)/torch.log(self.base.to(self.device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = STEFunction.apply(entropy_step)
        #print(entropy_step)

        means = []
        stds = []
        for b in range(bs):
            boundaries = torch.diff(entropy_step[b].type(torch.int64), prepend=~entropy_step[b][:1].type(torch.int64), append=~entropy_step[b][-1:].type(torch.int64))
            region_lengths = torch.diff(torch.nonzero(boundaries).squeeze())
            mean_region_length = region_lengths.float().mean()
            std_region_length = region_lengths.float().std()
            means.append(mean_region_length.item())
            stds.append(std_region_length.item())
        
        clst_sh = round(np.mean(means))
        q_row = q_row[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        q_col = q_col[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        v = v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        q_row = q_row.view(bs, q_row.shape[1]//clst_sh, clst_sh, feats)
        q_col = q_col.view(bs, q_col.shape[1]//clst_sh, clst_sh, feats)
        v = v.view(bs, v.shape[1]//clst_sh, clst_sh, feats)
        entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy_step = entropy_step[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
        q_row = (entropy*q_row).sum(-2)
        q_col = (entropy*q_col).sum(-2)
        v = (entropy*v).sum(-2)

        return q_row, q_col, v

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)