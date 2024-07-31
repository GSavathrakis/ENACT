import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
import matplotlib.pyplot as plt
#import ENACT

from torch.autograd.function import once_differentiable
from sklearn.cluster import KMeans

def kmeans(input, n_centroids):
    input_cpu = input.cpu().detach().numpy()
    bs, spat, ft = input_cpu.shape
    input_flatt = input_cpu.reshape(-1, ft)
    kmns = KMeans(n_clusters=n_centroids)
    kmns.fit(input_flatt)
    outp, inv_inds, cnt = np.unique(kmns.labels_, return_inverse=True, return_counts=True)
    clust_ids = cnt[inv_inds].reshape(bs, spat)
    prob_dist = clust_ids/spat

    return prob_dist



class ClustAttn(nn.Module):
    def __init__(self, sigma, d_model, dropout, n_heads, device):
        super().__init__()

        self.gaussian_kernel = (1./(sigma*torch.sqrt(torch.Tensor([2*np.pi]))))*torch.exp(-torch.pow(torch.arange(-(3*sigma-1),3*sigma), 2)/(2*torch.pow(torch.Tensor([sigma]),2)))
        self.Sobel_2der = torch.Tensor([-1., 2., -1.])
        self.base = torch.Tensor([2])
        
        self.W_q = nn.Linear(d_model, d_model)
        #self.dropout_q = nn.Dropout(dropout)
        self.W_k = nn.Linear(d_model, d_model)
        #self.dropout_k = nn.Dropout(dropout)
        self.W_v = nn.Linear(d_model, d_model)
        #self.dropout_v = nn.Dropout(dropout)
        self.W_o = nn.Linear(d_model, d_model)
        #self.dropout_o = nn.Dropout(dropout)

        #self.attn = nn.MultiheadAttention(d_model, n_heads)

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
        bs, spat, feats = q.shape
        prob_q = F.softmax(self.W_prob(q).squeeze(-1), -1) + 1e-8

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
        q = q[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        v = v[:,(spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2),:]
        q = q.view(bs, q.shape[1]//clst_sh, clst_sh, feats)
        v = v.view(bs, v.shape[1]//clst_sh, clst_sh, feats)
        entropy = entropy[:, (spat%clst_sh)//2:spat-(spat%clst_sh - (spat%clst_sh)//2)]
        entropy = F.softmax(entropy.view(bs, entropy.shape[1]//clst_sh, clst_sh), -1).unsqueeze(-1)
        q = (entropy*q).sum(-2)
        v = (entropy*v).sum(-2)

        q = self.W_q(q).view(bs, q.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        k = self.W_k(k).view(bs, spat, self.n_heads, feats//self.n_heads).permute(2,0,1,3)
        v = self.W_v(v).view(bs, v.shape[1], self.n_heads, feats//self.n_heads).permute(2,0,1,3)

        attention = self.W_o(torch.matmul(F.softmax(torch.matmul(k, q.transpose(2,3)), -1)/(feats//self.n_heads), v).permute(1,2,0,3).flatten(2,3)).permute(1,0,2)



        #attention = self.attn(k.permute(1,0,2), q.permute(1,0,2), v.permute(1,0,2))[0]
        
        """
        q = ENACT.enact_cluster(entropy, entropy_step, q)
        v  = ENACT.enact_cluster(entropy, entropy_step, v)

        q = self.W_q(torch.cat((q), dim=0).to(self.device))
        k = self.W_k(k)
        v  = self.W_v(torch.cat((v), dim=0).to(self.device))
        

        q = q.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2)
        k = k.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3)
        v  = v.view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2)

        #q = torch.randn(2, 9, 10).to(self.device)
        #k = torch.randn(2, 3, 4, 10).to(self.device)
        #v = torch.randn(2, 9, 10).to(self.device)

        n_clusters = ENACT.n_clusters(entropy_step)
        
        n_clusters_times_heads = n_clusters*self.n_heads
        n_clusters_times_heads_cumsum = np.array(n_clusters_times_heads).cumsum().tolist()
        end_inds_vals_cumsum = (np.array(n_clusters_times_heads).cumsum()-1).tolist()
        end_inds_attns = np.repeat(np.array(n_clusters_times_heads),spat*np.ones(np.array(n_clusters_times_heads).shape, dtype=np.int64)).tolist()
        end_inds_attns_cumsum = (np.array(end_inds_attns).cumsum() - 1).tolist()
        start_inds_attns_cumsum = (np.array(end_inds_attns).cumsum()).tolist()
        start_inds_attns_cumsum.insert(0,0)
        start_inds_attns_cumsum.pop()
        q_shapes_cumsum = np.array(n_clusters_times_heads).cumsum().tolist()
        end_inds_attns_all_pix = np.repeat(np.arange(len(np.array(end_inds_attns))), np.array(end_inds_attns)).tolist()
        start_inds_attns_cumsum_all_pixs = np.repeat(np.array(start_inds_attns_cumsum), np.array(end_inds_attns)).tolist()
        end_inds_vals_all_pix = np.repeat(np.arange(len(np.array(n_clusters_times_heads))), np.array(n_clusters_times_heads)).tolist()
        start_inds_vals_cumsum_all_pixs = (np.array(end_inds_vals_cumsum)+1).tolist()
        start_inds_vals_cumsum_all_pixs.insert(0,0)
        start_inds_vals_cumsum_all_pixs.pop()
        start_inds_vals_cumsum_all_pixs = np.repeat(np.array(start_inds_vals_cumsum_all_pixs), np.array(n_clusters_times_heads)).tolist()
        start_inds_vals_cumsum_all_pixs_all_inds = np.repeat(np.array(start_inds_vals_cumsum_all_pixs),spat*np.ones(np.array(start_inds_vals_cumsum_all_pixs).shape, dtype=np.int64)).tolist()
        gr_sizes = np.repeat(np.array(n_clusters_times_heads), np.array(n_clusters_times_heads))
        gr_sizes_all_pixs = np.repeat(end_inds_attns, end_inds_attns)
        
        
        
        
        n_clusters_times_heads_times_spat = np.repeat(np.array(n_clusters_times_heads),spat*np.ones(np.array(n_clusters_times_heads).shape, dtype=np.int64)).tolist()
        n_clusters_times_heads_times_spat_cumsum = np.array(n_clusters_times_heads_times_spat).cumsum().tolist()
        
        n_clusters = [spat, spat, spat, spat, spat, spat, spat, spat]

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.flatten(0,1).view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2)
        k = k.view(bs, spat, self.n_heads, feats//self.n_heads).permute(2, 0, 1, 3)
        v  = v.flatten(0,1).view(-1, self.n_heads, feats//self.n_heads).permute(1, 0, 2)
        
        

        #attention = ATTNFunction.apply(q, k, v, self.n_heads, end_inds_vals_cumsum, n_clusters_times_heads, q_shapes_cumsum, end_inds_attns, start_inds_attns_cumsum, end_inds_attns_all_pix, start_inds_attns_cumsum_all_pixs, end_inds_vals_all_pix, start_inds_vals_cumsum_all_pixs, gr_sizes, gr_sizes_all_pixs, start_inds_vals_cumsum_all_pixs_all_inds)
        
        attention = ATTNFunction.apply(q, k, v, self.n_heads, n_clusters, self.device)

        attention = attention.permute(1,2,0,3)
        attention = self.W_o(attention.flatten(2,3))
        """
        return attention






        """
        entropy_step[:,-1][entropy_step[:,-1]==entropy_step[:,-2]] = (~entropy_step[:,-1][entropy_step[:,-1]==entropy_step[:,-2]].type(torch.bool)).type(entropy_step.dtype)
        
        indices = torch.nonzero((entropy_step[:, 1:] != entropy_step[:, :-1]), as_tuple=False)
        indices[:, 1] += 1

        A = torch.zeros(indices.shape[0], spat).to(device)
        A[torch.arange(0,indices.shape[0]), indices[:,1]] = indices[:,0]+1.
        A = A[1:]+A[:-1]
        A[torch.sum(A, dim=-1)%2!=0] = A[torch.sum(A, dim=-1)%2!=0]//A[torch.sum(A, dim=-1)%2!=0].max(dim=-1).values.view(-1,1)*A[torch.sum(A, dim=-1)%2!=0].max(dim=-1).values.view(-1,1) \
                                    + torch.cat((A[torch.sum(A, dim=-1)%2!=0].max(dim=-1).values.view(-1,1),torch.zeros(A[torch.sum(A, dim=-1)%2!=0].max(dim=-1).values.shape[0],spat-1).to(device)),dim=1)
        A = torch.cat((A[0].view(1,-1), A), dim=0)
        A[0][torch.where(A[0]==1)[0][-1]] = 0
        A[0,0] = 1

        clust_sizes = torch.cat((torch.nonzero(A[:,0]).squeeze(), torch.tensor(A[:,0].shape[0]).view(1,).to(device)),dim=0)[1:] - torch.cat((torch.nonzero(A[:,0]).squeeze(), torch.tensor(A[:,0].shape[0]).view(1,).to(device)),dim=0)[:-1]
        
        A = torch.where(A != 0, 1, 0)
        A = torch.cumsum(A, dim=1)
        A = ((A == 1) | (A == 3)).type(torch.float)
        A[torch.cumsum(clust_sizes, dim=0)-1,-1]=1

        sz = entropy.size(0)
        entropy = self.non_zero_softmax(A*entropy[torch.repeat_interleave(torch.arange(sz).to(device), clust_sizes)]).unsqueeze(-1)
        qs = torch.split(torch.sum(entropy*q[torch.repeat_interleave(torch.arange(sz).to(device), clust_sizes)], dim=1), clust_sizes.tolist())
        vs = torch.split(torch.sum(entropy*v[torch.repeat_interleave(torch.arange(sz).to(device), clust_sizes)], dim=1), clust_sizes.tolist())

        pad = torch.split(torch.zeros(torch.sum(-1*clust_sizes+clust_sizes.max(dim=-1).values), feats).to(device), (-1*clust_sizes+clust_sizes.max(dim=-1).values).tolist())
        qs = torch.cat(list(map(lambda tensors: torch.cat(tensors, dim=0).unsqueeze(0), zip(qs, pad))), dim=0).permute(1,0,2)
        vs = torch.cat(list(map(lambda tensors: torch.cat(tensors, dim=0).unsqueeze(0), zip(vs, pad))), dim=0).permute(1,0,2)


        
        clust_sizes=[]
        qs=[]
        vs=[]
        for n in range(bs):
            curr_ind = torch.arange(0, spat-1).to(device)[(entropy_step[n,1:] != entropy_step[n,:-1])]
            if len(curr_ind) > 0:
                curr_ind = torch.cat((torch.tensor([0]).to(device), curr_ind + 1))
            
            
            if (curr_ind.shape[0]==0):
                break
            
            mask = torch.zeros(curr_ind.shape[0], spat).to(device)
            region_masks = torch.zeros_like(mask, dtype=torch.bool)
            region_masks[torch.arange(mask.size(0)), curr_ind] = True

            Ones = torch.ones(curr_ind.shape[0], spat).to(device)
            mask[region_masks] = Ones[region_masks]

            mask_start = (torch.cumsum(mask[1:] + mask[:-1], dim=1) == 1)
            mask_end = (torch.cumsum(mask[1:] + mask[:-1], dim=1) == 3)

            result = (mask_start | mask_end).type(torch.float)
            result[-1,-1]=1
            entropy_curr = result*entropy[n].unsqueeze(0).repeat(curr_ind.shape[0]-1, 1)
            qs.append(torch.sum(self.non_zero_softmax(entropy_curr).unsqueeze(-1)*q[n], dim=1))
            vs.append(torch.sum(self.non_zero_softmax(entropy_curr).unsqueeze(-1)*v[n], dim=1))
            clust_sizes.append(entropy_curr.shape[0])
            
        max_cl_size = max(clust_sizes)
        for n in range(bs):
            pad = torch.zeros(max_cl_size-qs[n].shape[0],qs[n].shape[1]).to(device)
            qs[n] = torch.cat((qs[n], pad),dim=0).unsqueeze(0)
            vs[n] = torch.cat((vs[n], pad),dim=0).unsqueeze(0)
            clust_qs, ks, clust_vs, n_heads, sizes, indices
        qs = torch.cat(qs, dim=0).permute(1,0,2)
        vs = torch.cat(vs, dim=0).permute(1,0,2)
        
        return qs, vs, clust_sizes
        
        if (curr_ind.shape[0]!=0):
            max_cl_size = max(clust_sizes)
            for n in range(bs):
                pad = torch.zeros(max_cl_size-qs[n].shape[0],qs[n].shape[1]).to(device)
                qs[n] = torch.cat((qs[n], pad),dim=0).unsqueeze(0)
                vs[n] = torch.cat((vs[n], pad),dim=0).unsqueeze(0)
            
            qs = torch.cat(qs, dim=0).permute(1,0,2)
            vs = torch.cat(vs, dim=0).permute(1,0,2)
        
            return qs, vs, True, clust_sizes
        else:
            return q.permute(1,0,2), v.permute(1,0,2), False, None
        """
        
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

"""
class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, clust_qs, ks, clust_vs, n_heads, n_clusters, device):
        output = ENACT.forward_mhsa(clust_qs, ks, clust_vs, n_heads, n_clusters)
        #output = output.to(device)
        ctx.n_heads = n_heads
        ctx.n_clusters = n_clusters
        ctx.device = device
        ctx.save_for_backward(clust_qs, ks, clust_vs)
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        clust_qs, ks, clust_vs = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_clusters = ctx.n_clusters
        device = ctx.device
        grad_qs, grad_ks, grad_vs = ENACT.backward_mhsa(clust_qs, ks, clust_vs, n_heads, n_clusters, grad_output)
        return grad_qs, grad_ks, grad_vs, None, None, None


class ATTNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, clust_qs, ks, clust_vs, n_heads, end_inds_vals_cumsum, n_clusters_times_heads, q_shapes_cumsum, end_inds_attns, start_inds_attns_cumsum, end_inds_attns_all_pix, start_inds_attns_cumsum_all_pixs, end_inds_vals_all_pix, start_inds_vals_cumsum_all_pixs, gr_sizes, gr_sizes_all_pixs, start_inds_vals_cumsum_all_pixs_all_inds):
        
        ctx.n_heads = n_heads
        ctx.end_inds_vals_cumsum = end_inds_vals_cumsum
        ctx.n_clusters_times_heads = n_clusters_times_heads
        ctx.q_shapes_cumsum = q_shapes_cumsum
        ctx.end_inds_attns = end_inds_attns
        ctx.start_inds_attns_cumsum = start_inds_attns_cumsum
        ctx.end_inds_attns_all_pix = end_inds_attns_all_pix
        ctx.start_inds_attns_cumsum_all_pixs = start_inds_attns_cumsum_all_pixs
        ctx.end_inds_vals_all_pix = end_inds_vals_all_pix
        ctx.start_inds_vals_cumsum_all_pixs = start_inds_vals_cumsum_all_pixs
        ctx.gr_sizes = gr_sizes
        ctx.gr_sizes_all_pixs = gr_sizes_all_pixs
        ctx.start_inds_vals_cumsum_all_pixs_all_inds = start_inds_vals_cumsum_all_pixs_all_inds
        print("start")
        

        output, attn_w = ENACT.forward_mhsa(clust_qs, ks, clust_vs, n_heads, end_inds_vals_cumsum, n_clusters_times_heads, q_shapes_cumsum, end_inds_attns, start_inds_attns_cumsum, end_inds_attns_all_pix, start_inds_attns_cumsum_all_pixs, end_inds_vals_all_pix, start_inds_vals_cumsum_all_pixs, gr_sizes)
        print("end")
        ctx.save_for_backward(clust_qs, ks, clust_vs, attn_w)
        #output = ENACT.forward_mhsa(clust_qs, ks, clust_vs, n_heads, )
        #ctx.save_for_backward(clust_qs, ks, clust_vs)
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        clust_qs, ks, clust_vs, attn_w = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_clusters_times_heads = ctx.n_clusters_times_heads
        end_inds_vals_cumsum = ctx.end_inds_vals_cumsum
        q_shapes_cumsum = ctx.q_shapes_cumsum
        end_inds_attns = ctx.end_inds_attns
        start_inds_attns_cumsum = ctx.start_inds_attns_cumsum
        end_inds_attns_all_pix = ctx.end_inds_attns_all_pix
        start_inds_attns_cumsum_all_pixs = ctx.start_inds_attns_cumsum_all_pixs
        end_inds_vals_all_pix = ctx.end_inds_vals_all_pix
        start_inds_vals_cumsum_all_pixs = ctx.start_inds_vals_cumsum_all_pixs
        gr_sizes = ctx.gr_sizes
        gr_sizes_all_pixs = ctx.gr_sizes_all_pixs
        start_inds_vals_cumsum_all_pixs_all_inds = ctx.start_inds_vals_cumsum_all_pixs_all_inds

        grad_qs, grad_ks, grad_vs = ENACT.backward_mhsa(grad_output, attn_w, clust_vs, ks, clust_qs, start_inds_vals_cumsum_all_pixs_all_inds, end_inds_attns_all_pix, gr_sizes_all_pixs, clust_qs.shape[1], end_inds_vals_cumsum, n_clusters_times_heads, q_shapes_cumsum, start_inds_attns_cumsum_all_pixs, gr_sizes_all_pixs, end_inds_vals_all_pix, start_inds_vals_cumsum_all_pixs)
        return grad_qs, grad_ks, grad_vs, None, None, None, None, None, None, None, None, None, None, None
"""
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)