import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import matplotlib.pyplot as plt


class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, dropout, n_heads, device):
        super().__init__()

        self.gaussian_kernel = (1./(sigma*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*sigma-1),3*sigma), 2)/(2*torch.pow(torch.Tensor([sigma]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.W_prob = nn.Linear(d_model, 1)

        self._reset_parameters()

        self.n_heads = n_heads
        self.device = device

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, q, k, v, h, w):
        q = q.permute(1,0,2) # New shape: BS x spatial x feature
        k = k.permute(1,0,2) # New shape: BS x spatial x feature
        v = v.permute(1,0,2) # New shape: BS x spatial x feature
        bs, spat, feats = k.shape
        prob_k = F.softmax(self.W_prob(k).squeeze(-1), -1) + 1e-8

        entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(self.device))
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
        k = k[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        v = v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        k = k.view(bs, k.shape[1]//clst_sh, clst_sh, feats)
        v = v.view(bs, v.shape[1]//clst_sh, clst_sh, feats)
        entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
        k = (entropy*k).sum(-2)
        v = (entropy*v).sum(-2)

        k = self.W_k(k).view(bs, k.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        q = self.W_q(q).view(bs, spat, self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        v = self.W_v(v).view(bs, v.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)

        attention = self.W_o(torch.matmul(F.softmax(torch.matmul(q, k.transpose(2,3)), -1)/(feats//self.n_heads), v).permute(1,2,0,3).flatten(2,3)).permute(1,0,2)

        return attention
        
    @staticmethod
    def plot_entropy(entropy, dims):
        if dims==1:
            spat = entropy.shape[0]
            plt.figure()
            plt.plot(np.linspace(0, spat, spat), entropy.cpu().detach().numpy())
            plt.xlabel("pixels")
            plt.ylabel("Smoothed information")
            plt.show()
        elif dims==2:
            plt.figure()
            plt.imshow(entropy[0].cpu().detach().numpy(), cmap='viridis')
            plt.colorbar()
            plt.show()
        else:
            print('Dimensions of entropy must be either 1d or 2d')

        
    
    @staticmethod
    def non_zero_softmax(tensor):
        tensor[tensor==0] = torch.tensor(-float('inf'))
        #tensor = torch.exp(tensor)/torch.sum(torch.exp(tensor),dim=-1).unsqueeze(-1).expand(tensor.shape)
        return F.softmax(tensor) 
    
    @staticmethod
    def gaussian_2d(sx, sy, range_x, range_y, device):
        x = torch.arange(-range_x//2, range_x//2+1).to(device)
        y = torch.arange(-range_y//2, range_y//2+1).to(device)

        YY, XX = torch.meshgrid(y, x)

        gaussian_kernel = 1./(2*torch.Tensor([np.pi]).to(device)*sx*sy)*torch.exp(-torch.pow(XX,2)/(2*sx**2)-torch.pow(YY,2)/(2*sy**2))

        return gaussian_kernel

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)