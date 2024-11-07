import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

import copy
import ENACT
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
        
        k = ENACT.enact_cluster(entropy, entropy_step, k)
        v  = ENACT.enact_cluster(entropy, entropy_step, v)

        q = self.W_q(q)
        k = self.W_k(torch.cat((k), dim=0).to(self.device))
        v  = self.W_v(torch.cat((v), dim=0).to(self.device))
        

        q = q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3)
        k = k.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
        v  = v.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)

        n_clusters = ENACT.n_clusters(entropy_step)

        sizes = np.array(n_clusters*self.n_heads).cumsum().tolist()
        start_inds = copy.deepcopy(sizes)
        start_inds.insert(0,0)
        start_inds.pop()
        
        attention = ATTNFunction.apply(q, k, v, start_inds, sizes)

        attention = attention.permute(1,2,0,3)
        attention = attention.flatten(2,3)
        attention = attention.permute(1,0,2)
        attention = self.W_o(attention)

        
        
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


class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, clust_ks, clust_vs, start_indices, cl_sizes):
        
        ctx.start_indices = start_indices
        ctx.cl_sizes = cl_sizes
        output, attn_w = ENACT.forward_mhsa(qs, clust_ks, clust_vs, start_indices, cl_sizes)
        print(output)
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()
        ctx.save_for_backward(qs, clust_ks, clust_vs, attn_w)
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        
        qs, clust_ks, clust_vs, attn_w = ctx.saved_tensors
        start_indices = ctx.start_indices
        cl_sizes = ctx.cl_sizes

        grad_qs, grad_ks, grad_vs, grad_soft_attn_ws, grad_attn_ws = ENACT.backward_mhsa(grad_output, attn_w, qs, clust_ks, clust_vs, start_indices, cl_sizes)
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()
        return grad_qs, grad_ks, grad_vs, None, None

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)