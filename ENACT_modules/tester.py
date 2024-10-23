import torch 
import ENACT

tens1 = torch.randn(2,8,784,32).to('cuda')
tens2 = torch.randn(138, 32).to('cuda')
tens3 = torch.randn(138, 32).to('cuda')

start_inds = [0, 15, 27, 35, 44, 53, 67, 89, 93, 97, 105, 112, 117, 121, 128, 135]
n_clusts = [15, 12, 8, 9, 9, 14, 12, 4, 4, 8, 7, 5, 4, 7, 7, 3]

res, ws = ENACT.forward_mhsa(tens1, tens2, tens3, start_inds, n_clusts)
print(ws.shape)
print(ws.permute(1,0,2).sum(dim=-2))