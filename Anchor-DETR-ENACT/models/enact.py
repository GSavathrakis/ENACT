import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

import numpy as np
import matplotlib.pyplot as plt
import ENACT
import copy

class CLUSTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k_rows, k_cols, v, ent, ent_step, st_inds, reg_l):
        ctx.ent_step = ent_step
        ctx.st_inds = st_inds
        ctx.reg_l = reg_l
        
        k_rows_cl, k_cols_cl, v_cl = ENACT.enact_cluster_forward_rcda(k_rows, k_cols, v, ent, ent_step, st_inds, reg_l)
        
        ctx.save_for_backward(k_rows, k_cols, v, ent)

        return k_rows_cl, k_cols_cl, v_cl
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_k_rows_cl, grad_k_cols_cl, grad_v_cl):
        k_rows, k_cols, v, ent = ctx.saved_tensors
        ent_step = ctx.ent_step
        st_inds = ctx.st_inds
        reg_l = ctx.reg_l
        
        k_rows, k_cols, v, ent = ENACT.enact_cluster_backward_rcda(grad_k_rows_cl, grad_k_cols_cl, grad_v_cl, k_rows, k_cols, v, ent, ent_step, st_inds, reg_l)
        ent = ent.sum(-1)
        
        return k_rows, k_cols, v, ent, None, None, None

class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, n_heads, device):
        super().__init__()

        self.gaussian_kernel = (1./(sigma*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*sigma-1),3*sigma), 2)/(2*torch.pow(torch.Tensor([sigma]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])

        self.W_prob = nn.Linear(d_model, 1)

        self._reset_parameters()

        self.device = device
        self.n_heads = n_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, k_row, k_col, v):
        bs, spat, feats = k_row.shape
        
        prob_k = F.softmax(self.W_prob(k_row).squeeze(-1), -1) + 1e-8

        entropy = -prob_k*torch.log(prob_k)/torch.log(self.base.to(self.device))
        entropy = F.conv1d(entropy.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step = F.conv1d(entropy.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step = (entropy_step > 0).to(torch.float)
        entropy_step = (entropy_step*2-1).to(torch.int)

        entropy_step = entropy_step.flatten(0,1)
        entropy = entropy.flatten(0,1)
        k_row = k_row.flatten(0,1)
        k_col = k_col.flatten(0,1)
        v = v.flatten(0,1)

        start_indices = torch.sign(entropy_step)  # Convert elements to +1 or -1 based on their sign
        start_indices = start_indices[1:] != start_indices[:-1]  # Identify where sign changes
        start_indices = torch.cat((torch.tensor([0]).to(self.device), torch.nonzero(start_indices, as_tuple=True)[0] + 1))
        start_indices = torch.unique(torch.sort(torch.cat(((torch.Tensor([spat]*(bs-1))*torch.linspace(1,bs-1,bs-1)).to(self.device), start_indices)))[0])
        region_lengths = torch.diff(torch.cat((start_indices, torch.tensor([entropy_step.size(0)]).to(self.device))))
        
        
        start_indices = start_indices.to(torch.int)
        region_lengths = region_lengths.to(torch.int)

        k_row_cl, k_col_cl, v_cl = CLUSTFunction.apply(k_row, k_col, v, entropy, entropy_step, start_indices, region_lengths)

        region_lengths = torch.cumsum(region_lengths, dim=0)
        region_lengths = torch.where(region_lengths%spat==0)
        region_lengths = torch.diff(torch.cat((torch.tensor([0]).to('cuda'), region_lengths[0]+1)))

        region_lengths = region_lengths.repeat(self.n_heads).to(torch.int)
        start_indices = copy.deepcopy(torch.cumsum(region_lengths, 0))
        start_indices = torch.cat((torch.tensor([0]).to(self.device), start_indices))#start_inds.insert(0,0)
        start_indices = start_indices[:-1].to(torch.int)
        """

        prob_k_row = F.softmax(self.W_prob(k_row).squeeze(-1), -1) + 1e-8

        entropy_row = -prob_k_row*torch.log(prob_k_row)/torch.log(self.base.to(self.device))
        entropy_row = F.conv1d(entropy_row.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step_row = F.conv1d(entropy_row.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step_row = STEFunction.apply(entropy_step_row)

        prob_k_col = F.softmax(self.W_prob(k_col).squeeze(-1), -1) + 1e-8

        entropy_col = -prob_k_col*torch.log(prob_k_col)/torch.log(self.base.to(self.device))
        entropy_col = F.conv1d(entropy_col.unsqueeze(1), self.gaussian_kernel.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        
        entropy_step_col = F.conv1d(entropy_col.unsqueeze(1), self.Sobel_2der.to(self.device).unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)
        entropy_step_col = STEFunction.apply(entropy_step_col)

        means = (~torch.logical_xor(entropy_step_row, entropy_step_col)).type(torch.float64)
        print(means.mean(-1))
        #print(entropy_step)

        entropy_step = entropy_step_row
        entropy = entropy_row
        

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
        k_row = k_row[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        k_col = k_col[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        v = v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        k_row = k_row.view(bs, k_row.shape[1]//clst_sh, clst_sh, feats)
        k_col = k_col.view(bs, k_col.shape[1]//clst_sh, clst_sh, feats)
        v = v.view(bs, v.shape[1]//clst_sh, clst_sh, feats)
        entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy_step = entropy_step[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
        k_row = (entropy*k_row).sum(-2)
        k_col = (entropy*k_col).sum(-2)
        v = (entropy*v).sum(-2)
        """

        return k_row_cl, k_col_cl, v_cl, start_indices, region_lengths