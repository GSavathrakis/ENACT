import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

import copy
import ENACT
import numpy as np
import matplotlib.pyplot as plt
import time

#torch.autograd.set_detect_anomaly(True)


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
        start_time = time.time()
        prob_k = F.softmax(self.W_prob(k).squeeze(-1), -1) + 1e-8

        entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(self.device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1).to('cpu')
        entropy_step = (entropy_step > 0).to(torch.float32)
        entropy_step = (entropy_step*2-1).to(torch.int32)

        entropy_step = entropy_step.flatten(0,1)
        entropy = entropy.flatten(0,1)
        k = k.flatten(0,1)
        v = v.flatten(0,1)

        aux = torch.sign(entropy_step)  # Convert elements to +1 or -1 based on their sign
        aux = aux[1:] != aux[:-1]  # Identify where sign changes
        start_indices = torch.cat((torch.tensor([0]), torch.nonzero(aux, as_tuple=True)[0] + 1))
        start_indices = torch.unique(torch.sort(torch.cat((torch.Tensor([spat]*(bs-1))*torch.linspace(1,bs-1,bs-1), start_indices)))[0])
        region_lengths = torch.diff(torch.cat((start_indices, torch.tensor([entropy_step.size(0)]))))

        entropy_step = entropy_step.to(torch.int32).tolist()
        start_indices = start_indices.to(torch.int32).tolist()
        region_lengths = region_lengths.to(torch.int32).tolist()

        sizes = np.array(region_lengths*self.n_heads)
        start_inds = copy.deepcopy(sizes.cumsum().tolist())
        start_inds.insert(0,0)
        start_inds.pop()
        sizes = sizes.tolist()
        
        
        k_cl, v_cl = CLUSTFunction.apply(k, v, entropy, entropy_step, start_indices, region_lengths)
        
        """
        k = k.flatten(0,1)
        v = v.flatten(0,1)
        
        sizes = np.array([spat]*bs*self.n_heads)
        start_inds = copy.deepcopy(sizes.cumsum().tolist())
        start_inds.insert(0,0)
        start_inds.pop()
        sizes = sizes.tolist()
        """

        q = self.W_q(q)
        k_cl = self.W_k(k_cl)
        v_cl  = self.W_v(v_cl)
        
        q = q.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3)
        k_cl = k_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
        v_cl = v_cl.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2).flatten(0,1)
            
        attention = ATTNFunction.apply(q, k_cl, v_cl, start_inds, sizes)
        

        attention = attention.permute(1,2,0,3)
        attention = attention.flatten(2,3)
        attention = attention.permute(1,0,2)
        attention = self.W_o(attention)
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
        ctx.save_for_backward(k, v, ent)
        k_cl, v_cl = ENACT.enact_cluster_forward(k, v, ent, ent_step, st_inds, reg_l)
        #torch.cuda.synchronize()
        return k_cl, v_cl
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_k_cl, grad_v_cl):
        k, v, ent = ctx.saved_tensors
        ent_step = ctx.ent_step
        st_inds = ctx.st_inds
        reg_l = ctx.reg_l
        grad_k, grad_v, grad_entr = ENACT.enact_cluster_backward(grad_k_cl, grad_v_cl, k, v, ent, ent_step, st_inds, reg_l)
        #torch.cuda.synchronize()
        return grad_k, grad_v, grad_entr, None, None, None

class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, clust_ks, clust_vs, start_indices, cl_sizes):
        
        ctx.start_indices = start_indices
        ctx.cl_sizes = cl_sizes
        
        output, attn_w = ENACT.forward_mhsa(qs, clust_ks, clust_vs, start_indices, cl_sizes)
        #torch.cuda.synchronize()
        ctx.save_for_backward(qs, clust_ks, clust_vs, attn_w)

        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        
        qs, clust_ks, clust_vs, attn_w = ctx.saved_tensors
        start_indices = ctx.start_indices
        cl_sizes = ctx.cl_sizes
        #start_time = time.time()
        grad_qs, grad_ks, grad_vs = ENACT.backward_mhsa(grad_output, attn_w, qs, clust_ks, clust_vs, start_indices, cl_sizes)
        #torch.cuda.synchronize()
        #end_time = time.time()
        #print(f"Time lapsed backward:{end_time-start_time}")
        return grad_qs, grad_ks, grad_vs, None, None