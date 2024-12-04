#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void soft(const torch::Tensor attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int num_heads, const int batch_size, 
          const int spatial_dims, const int concat_spatial_dims, torch::Tensor soft_attn_ws){

    int n_threads_soft_attn_ws = 1024;

    int n_blocks_soft_attn_ws_x = (spatial_dims + n_threads_soft_attn_ws - 1)/n_threads_soft_attn_ws + 1;
    int n_blocks_soft_attn_ws_y = num_heads*batch_size + 1;
    
    dim3 numBlocks_soft_attn_ws(n_blocks_soft_attn_ws_x, n_blocks_soft_attn_ws_y);
    dim3 threadsPerBlock_soft_attn_ws(n_threads_soft_attn_ws, 1);
    softmax<<<numBlocks_soft_attn_ws, threadsPerBlock_soft_attn_ws>>>(attn_ws.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
}