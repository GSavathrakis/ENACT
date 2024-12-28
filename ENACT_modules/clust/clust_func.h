#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

vector<torch::Tensor> enact_cluster_forward(torch::Tensor Keys, torch::Tensor Values, torch::Tensor Entropy, torch::Tensor Entropy_step, torch::Tensor start_inds, torch::Tensor region_lengths);
vector<torch::Tensor> enact_cluster_backward(torch::Tensor grad_Keys_cl, torch::Tensor grad_Values_cl, torch::Tensor Keys, torch::Tensor Values, torch::Tensor Entropy, torch::Tensor Entropy_step, torch::Tensor start_inds, torch::Tensor region_lengths);
vector<torch::Tensor> enact_cluster_forward_rcda(torch::Tensor Keys_rows, torch::Tensor Keys_cols, torch::Tensor Values, torch::Tensor Entropy, torch::Tensor Entropy_step, torch::Tensor start_inds, torch::Tensor region_lengths);
vector<torch::Tensor> enact_cluster_backward_rcda(torch::Tensor grad_Keys_rows_cl, torch::Tensor grad_Keys_cols_cl, torch::Tensor grad_Values_cl, torch::Tensor Keys_rows, torch::Tensor Keys_cols, torch::Tensor Values, torch::Tensor Entropy, torch::Tensor Entropy_step, torch::Tensor start_inds, torch::Tensor region_lengths);

#endif // CLUSTERS_H