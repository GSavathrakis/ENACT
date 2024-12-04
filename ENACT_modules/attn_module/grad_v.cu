#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void grad_values(const torch::Tensor grad_output, const torch::Tensor soft_attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
                 const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_values){
    
    int n_threads_grad_v_x = 32;
    int n_threads_grad_v_y = 32;

    int n_blocks_grad_v_x = (concat_spatial_dims + n_threads_grad_v_x - 1)/n_threads_grad_v_x + 1;
    int n_blocks_grad_v_y = (feature_dims + n_threads_grad_v_y - 1)/n_threads_grad_v_y + 1;
    int batch_grad_v = num_heads*batch_size + 1;

    dim3 numBlocks_grad_v(n_blocks_grad_v_x, n_blocks_grad_v_y, batch_grad_v);
    dim3 threadsPerBlock_grad_v(n_threads_grad_v_x, n_threads_grad_v_y);
    grad_v<<<numBlocks_grad_v, threadsPerBlock_grad_v>>>(soft_attn_ws.data_ptr<float>(), grad_output.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, spatial_dims, feature_dims, grad_values.data_ptr<float>());
    cudaDeviceSynchronize();

}