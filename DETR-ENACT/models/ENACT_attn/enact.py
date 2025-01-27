import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

import copy
import ENACT
import numpy as np
import matplotlib.pyplot as plt


class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, n_heads, device):
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
    
    def forward(self, q, k, v, h, w, n_layer):
        q = q.permute(1,0,2) # New shape: BS x spatial x feature
        k = k.permute(1,0,2) # New shape: BS x spatial x feature
        v = v.permute(1,0,2) # New shape: BS x spatial x feature
        bs, spat, feats = k.shape

        entropy = F.softmax(self.W_prob(k).squeeze(-1), -1) + 1e-8

        entropy = -entropy*torch.log(entropy)/torch.log(self.base.to(self.device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = ((entropy_step > 0).to(torch.int))*2-1
        #entropy_step = (entropy_step*2-1).to(torch.int)

        entropy_step = entropy_step.flatten(0,1)
        entropy = entropy.flatten(0,1)
        k = k.flatten(0,1)
        v = v.flatten(0,1)

        
        start_indices = entropy_step[1:] != entropy_step[:-1]  # Identify where sign changes
        start_indices = torch.cat((torch.tensor([0]).to(self.device), torch.nonzero(start_indices, as_tuple=True)[0] + 1))
        start_indices = torch.unique(torch.sort(torch.cat(((torch.Tensor([spat]*(bs-1))*torch.linspace(1,bs-1,bs-1)).to(self.device), start_indices)))[0])
        region_lengths = torch.diff(torch.cat((start_indices, torch.tensor([entropy_step.size(0)]).to(self.device))))
        
        
        start_indices = start_indices.to(torch.int)
        region_lengths = region_lengths.to(torch.int)

        k_cl, v_cl = CLUSTFunction.apply(k, v, entropy, entropy_step, start_indices, region_lengths)

        region_lengths = torch.cumsum(region_lengths, dim=0)
        region_lengths = torch.where(region_lengths%spat==0)
        region_lengths = torch.diff(torch.cat((torch.tensor([0]).to('cuda'), region_lengths[0]+1)))

        region_lengths = region_lengths.repeat(self.n_heads).to(torch.int)
        start_indices = copy.deepcopy(torch.cumsum(region_lengths, 0))
        start_indices = torch.cat((torch.tensor([0]).to(self.device), start_indices))#start_inds.insert(0,0)
        start_indices = start_indices[:-1].to(torch.int)

        q = self.W_q(q)
        k_cl = self.W_k(k_cl)
        v_cl  = self.W_v(v_cl)

        q = q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3).flatten(0,1)
        k_cl = k_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
        v_cl = v_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
            
        attention = ATTNFunction.apply(q, k_cl, v_cl, start_indices, region_lengths)

        attention = attention.view(self.n_heads, bs, spat, feats//self.n_heads)
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

class CLUSTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, ent, ent_step, st_inds, reg_l):
        ctx.ent_step = ent_step
        ctx.st_inds = st_inds
        ctx.reg_l = reg_l
        
        k_cl, v_cl = ENACT.enact_cluster_forward(k, v, ent, ent_step, st_inds, reg_l)
        
        ctx.save_for_backward(k, v, ent)

        return k_cl, v_cl
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_k_cl, grad_v_cl):
        k, v, ent = ctx.saved_tensors
        ent_step = ctx.ent_step
        st_inds = ctx.st_inds
        reg_l = ctx.reg_l
        
        k, v, ent = ENACT.enact_cluster_backward(grad_k_cl, grad_v_cl, k, v, ent, ent_step, st_inds, reg_l)
        ent = ent.sum(-1)
        
        return k, v, ent, None, None, None

class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, clust_ks, clust_vs, start_indices, cl_sizes):
        
        ctx.start_indices = start_indices
        ctx.cl_sizes = cl_sizes

        output, soft_attn_w = ENACT.forward_mhsa(qs, clust_ks, clust_vs, start_indices, cl_sizes)
        
        ctx.save_for_backward(qs, clust_ks, clust_vs, soft_attn_w)

        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        
        qs, clust_ks, clust_vs, soft_attn_w = ctx.saved_tensors
        start_indices = ctx.start_indices
        cl_sizes = ctx.cl_sizes

        qs, clust_ks, clust_vs = ENACT.backward_mhsa(grad_output, soft_attn_w, qs, clust_ks, clust_vs, start_indices, cl_sizes)
        
        return qs, clust_ks, clust_vs, None, None