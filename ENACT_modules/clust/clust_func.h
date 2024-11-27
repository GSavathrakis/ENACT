#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <torch/torch.h>
using namespace std;

vector<at::Tensor> enact_cluster_forward(at::Tensor Keys, at::Tensor Values, at::Tensor Entropy, vector<int> Entropy_step, vector<int> start_inds, vector<int> region_lengths);
vector<at::Tensor> enact_cluster_backward(at::Tensor grad_Keys_cl, at::Tensor grad_Values_cl, at::Tensor Keys, at::Tensor Values, at::Tensor Entropy, vector<int> Entropy_step, vector<int> start_inds, vector<int> region_lengths);

#endif // CLUSTERS_H