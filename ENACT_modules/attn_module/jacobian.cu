#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void Jacobian(const torch::Tensor grad_soft_attn_ws, const torch::Tensor soft_attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
              const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, torch::Tensor grad_attn_ws){

    int n_threads_grad_attn_w_x = 32;
    int n_threads_grad_attn_w_y = 32;

    int n_blocks_grad_attn_w_x = (concat_spatial_dims + n_threads_grad_attn_w_x - 1)/n_threads_grad_attn_w_x + 1;
    int n_blocks_grad_attn_w_y = (spatial_dims + n_threads_grad_attn_w_y - 1)/n_threads_grad_attn_w_y + 1;
    int batch_grad_attn_w = num_heads*batch_size + 1;

    dim3 numBlocks_grad_attn_w(n_blocks_grad_attn_w_x, n_blocks_grad_attn_w_y, batch_grad_attn_w);
    dim3 threadsPerBlock_grad_attn_w(n_threads_grad_attn_w_x, n_threads_grad_attn_w_y);
    grad_attn_w<<<numBlocks_grad_attn_w, threadsPerBlock_grad_attn_w>>>(grad_soft_attn_ws.data_ptr<float>(), soft_attn_ws.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, grad_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
}