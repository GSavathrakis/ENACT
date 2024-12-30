#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

vector<torch::Tensor> forward_mhsa(const torch::Tensor Queries, const torch::Tensor Keys, const torch::Tensor Values, const torch::Tensor clust_start_inds, 
                  const torch::Tensor clust_sizes);
vector<torch::Tensor> backward_mhsa(const torch::Tensor grad_output, const torch::Tensor soft_attn_ws, const torch::Tensor Queries, const torch::Tensor Keys, 
                   const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes);
torch::Tensor forward_rcda_w(const torch::Tensor Q, const torch::Tensor K, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes);
torch::Tensor forward_rcda_map(const torch::Tensor weights, const torch::Tensor V, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int nh_bs);
vector<torch::Tensor> backward_rcda_map(const torch::Tensor grad_output, const torch::Tensor weights, const torch::Tensor V, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int nh_bs);
vector<torch::Tensor> backward_rcda_w(const torch::Tensor grad_w, const torch::Tensor w, const torch::Tensor Q, const torch::Tensor K, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes);

#endif // FUNCTIONS_H