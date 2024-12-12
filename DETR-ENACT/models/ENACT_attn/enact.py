import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

import copy
import ENACT
import numpy as np
import matplotlib.pyplot as plt
import time
import math

#torch.autograd.set_detect_anomaly(True)


class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, dropout, n_heads, device):
        super().__init__()

        self.gaussian_kernel = (1./(sigma*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*sigma-1),3*sigma), 2)/(2*torch.pow(torch.Tensor([sigma]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])

        #self.conv_q = nn.Conv2d(d_model, d_model, )
        
        self.W_q = nn.Linear(d_model//n_heads, d_model//n_heads)
        self.W_k = nn.Linear(d_model//n_heads, d_model//n_heads)
        self.W_v = nn.Linear(d_model//n_heads, d_model//n_heads)
        self.W_o = nn.Linear(d_model//n_heads, d_model//n_heads)

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
        start_time = time.time()
        prob_k = F.softmax(self.W_prob(k).squeeze(-1), -1) + 1e-8

        entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(self.device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = (entropy_step > 0).to(torch.float32)
        entropy_step = (entropy_step*2-1).to(torch.int32)

        entropy_step = entropy_step.flatten(0,1)
        entropy = entropy.flatten(0,1)
        k = k.flatten(0,1)
        v = v.flatten(0,1)

        aux = torch.sign(entropy_step)  # Convert elements to +1 or -1 based on their sign
        aux = aux[1:] != aux[:-1]  # Identify where sign changes
        start_indices = torch.cat((torch.tensor([0]).to(self.device), torch.nonzero(aux, as_tuple=True)[0] + 1))
        start_indices = torch.unique(torch.sort(torch.cat(((torch.Tensor([spat]*(bs-1))*torch.linspace(1,bs-1,bs-1)).to(self.device), start_indices)))[0])
        region_lengths = torch.diff(torch.cat((start_indices, torch.tensor([entropy_step.size(0)]).to(self.device))))
        
        
        start_indices = start_indices.to(torch.int32)
        region_lengths = region_lengths.to(torch.int32)

        k_cl, v_cl = CLUSTFunction.apply(k, v, entropy, entropy_step, start_indices, region_lengths)

        n_clusts = torch.cumsum(region_lengths, dim=0)
        n_clusts = torch.where(n_clusts%spat==0)
        n_clusts = torch.diff(torch.cat((torch.tensor([0]).to('cuda'), n_clusts[0]+1)))

        n_clusts = n_clusts.repeat(self.n_heads).to(torch.int32)
        start_inds = copy.deepcopy(torch.cumsum(n_clusts, 0))
        start_inds = torch.cat((torch.tensor([0]).to(self.device), start_inds))#start_inds.insert(0,0)
        start_inds = start_inds[:-1].to(torch.int32)


        
        """
        k = k.flatten(0,1)
        v = v.flatten(0,1)
        
        sizes = np.array([spat]*bs*self.n_heads)
        start_inds = copy.deepcopy(sizes.cumsum().tolist())
        start_inds.insert(0,0)
        start_inds.pop()
        sizes = sizes.tolist()
        """

        q = q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3).flatten(0,1)
        k_cl = k_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
        v_cl = v_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)

        q = self.W_q(q)
        k_cl = self.W_k(k_cl)
        v_cl  = self.W_v(v_cl)
            
        attention = ATTNFunction.apply(q, k_cl, v_cl, start_inds, n_clusts)

        attention = attention.view(self.n_heads, bs, spat, feats//self.n_heads)
        attention = self.W_o(attention)
        attention = attention.permute(1,2,0,3)
        attention = attention.flatten(2,3)
        attention = attention.permute(1,0,2)
        end_time = time.time()
        #print(f"Time lapsed forward:{end_time-start_time}")

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
        if torch.isnan(k_cl).any():
            print("NaN detected in k_cl")
        if torch.isnan(v_cl).any():
            print("NaN detected in v_cl")
        #torch.cuda.synchronize()
        ctx.save_for_backward(k, v, ent)

        return k_cl, v_cl
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_k_cl, grad_v_cl):
        k, v, ent = ctx.saved_tensors
        ent_step = ctx.ent_step
        st_inds = ctx.st_inds
        reg_l = ctx.reg_l
        if torch.isnan(grad_k_cl).any():
            print("NaN detected in grad_k_cl")
        if torch.isnan(grad_v_cl).any():
            print("NaN detected in grad_v_cl")
        grad_k, grad_v, grad_entr = ENACT.enact_cluster_backward(grad_k_cl, grad_v_cl, k, v, ent, ent_step, st_inds, reg_l)
        grad_entr = grad_entr.sum(-1)
        #torch.cuda.synchronize()
        return grad_k, grad_v, grad_entr, None, None, None

class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, clust_ks, clust_vs, start_indices, cl_sizes):
        
        ctx.start_indices = start_indices
        ctx.cl_sizes = cl_sizes
        if torch.isnan(qs).any():
            print("NaN detected in qs")
        if torch.isnan(clust_ks).any():
            print("NaN detected in clust_ks")
        if torch.isnan(clust_vs).any():
            print("NaN detected in clust_vs")

        output, soft_attn_w = ENACT.forward_mhsa(qs, clust_ks, clust_vs, start_indices, cl_sizes)
        """
        prod = torch.matmul(qs[6], clust_ks[start_indices[6]:start_indices[6]+cl_sizes[6]].transpose(0,1))/math.sqrt(qs.shape[2])
        test_attn_w = F.softmax(prod, -1)
        print('-----------------')
        print(soft_attn_w[:,start_indices[6]:start_indices[6]+cl_sizes[6]])
        print(test_attn_w)
        if torch.isnan(soft_attn_w).any():
            print("NaN detected in soft_attn_w")
        if torch.isnan(output).any():
            print("NaN detected in output")
        """
        
        #torch.cuda.synchronize()
        ctx.save_for_backward(qs, clust_ks, clust_vs, soft_attn_w)

        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        
        qs, clust_ks, clust_vs, soft_attn_w = ctx.saved_tensors
        start_indices = ctx.start_indices
        cl_sizes = ctx.cl_sizes

        """
        num_heads = grad_output.shape[0]
        batch_size = grad_output.shape[1]
        spatial_dims = grad_output.shape[2]
        feature_dims = grad_output.shape[3]
        concat_spatial_dims = clust_ks.shape[0]

        if torch.isnan(grad_output).any():
            print("NaN detected in grad_output")
        if torch.isnan(soft_attn_w).any():
            print("NaN detected in soft_attn_w")
        
        grad_soft_attn_w = torch.zeros(soft_attn_w.shape, dtype=torch.float, device=qs.device)
        grad_attn_w = torch.zeros(soft_attn_w.shape, dtype=torch.float, device=qs.device)
        grad_qs = torch.zeros(qs.shape, dtype=torch.float, device=qs.device)
        grad_ks = torch.zeros(clust_ks.shape, dtype=torch.float, device=qs.device)
        grad_vs = torch.zeros(clust_vs.shape, dtype=torch.float, device=qs.device)

        ENACT.grad_Values(grad_output, soft_attn_w, start_indices, cl_sizes, num_heads, batch_size, spatial_dims, concat_spatial_dims, feature_dims, grad_vs)
        ENACT.grad_Attention(grad_output, clust_vs, start_indices, cl_sizes, num_heads, batch_size, spatial_dims, concat_spatial_dims, feature_dims, grad_soft_attn_w)
        ENACT.Jacobian(grad_soft_attn_w, soft_attn_w, start_indices, cl_sizes, num_heads, batch_size, spatial_dims, concat_spatial_dims, grad_attn_w)
        ENACT.grad_Queries(grad_attn_w, clust_ks, start_indices, cl_sizes, num_heads, batch_size, spatial_dims, concat_spatial_dims, feature_dims, grad_qs)
        ENACT.grad_Keys(grad_attn_w, qs, start_indices, cl_sizes, num_heads, batch_size, spatial_dims, concat_spatial_dims, feature_dims, grad_ks)
        """
        if torch.isnan(grad_output).any():
            print("NaN detected in grad_output")
        grad_qs, grad_ks, grad_vs = ENACT.backward_mhsa(grad_output, soft_attn_w, qs, clust_ks, clust_vs, start_indices, cl_sizes)
        if torch.isnan(grad_qs).any():
            print("NaN detected in grad_qs")
        if torch.isnan(grad_ks).any():
            print("NaN detected in grad_ks")
        if torch.isnan(grad_vs).any():
            print("NaN detected in grad_ks")
        #torch.cuda.synchronize()
        
        #print(f"Time lapsed backward:{end_time-start_time}")
        return grad_qs, grad_ks, grad_vs, None, None