#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void grad_Keys(const torch::Tensor grad_attn_ws, const torch::Tensor Queries, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
               const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_keys){

    int n_threads_grad_k_x = 32;
    int n_threads_grad_k_y = 32;

    int n_blocks_grad_k_x = (concat_spatial_dims + n_threads_grad_k_x - 1)/n_threads_grad_k_x + 1;
    int n_blocks_grad_k_y = (feature_dims + n_threads_grad_k_y - 1)/n_threads_grad_k_y + 1;
    int batch_grad_k = num_heads*batch_size + 1;

    dim3 numBlocks_grad_k(n_blocks_grad_k_x, n_blocks_grad_k_y, batch_grad_k);
    dim3 threadsPerBlock_grad_k(n_threads_grad_k_x, n_threads_grad_k_y);
    grad_k<<<numBlocks_grad_k, threadsPerBlock_grad_k>>>(grad_attn_ws.data_ptr<float>(), Queries.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, spatial_dims, grad_keys.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_keys = grad_keys/sqrt(feature_dims);

}