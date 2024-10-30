#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

vector<at::Tensor> forward_mhsa(at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes);
vector<at::Tensor> backward_mhsa(at::Tensor grad_output, at::Tensor soft_attn_ws, at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes);

#endif // FUNCTIONS_H