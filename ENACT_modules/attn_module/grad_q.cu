#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void grad_Queries(const torch::Tensor grad_attn_ws, const torch::Tensor Keys, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
                  const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_queries){

    int n_threads_grad_q_x = 32;
    int n_threads_grad_q_y = 32;

    int n_blocks_grad_q_x = (spatial_dims + n_threads_grad_q_x - 1)/n_threads_grad_q_x + 1;
    int n_blocks_grad_q_y = (feature_dims + n_threads_grad_q_y - 1)/n_threads_grad_q_y + 1;
    int batch_grad_q = num_heads*batch_size + 1;

    dim3 numBlocks_grad_q(n_blocks_grad_q_x, n_blocks_grad_q_y, batch_grad_q);
    dim3 threadsPerBlock_grad_q(n_threads_grad_q_x, n_threads_grad_q_y);
    grad_q<<<numBlocks_grad_q, threadsPerBlock_grad_q>>>(grad_attn_ws.data_ptr<float>(), Keys.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, feature_dims, grad_queries.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_queries = grad_queries/sqrt(feature_dims);
}