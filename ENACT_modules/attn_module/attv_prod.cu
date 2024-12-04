#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void attn(const torch::Tensor soft_attn_ws, const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
          const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor attn){
    
    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (spatial_dims + n_threads_attn_x - 1)/n_threads_attn_x + 1;
    int n_blocks_attn_y = (feature_dims + n_threads_attn_y - 1)/n_threads_attn_y + 1;
    int batch_attn = num_heads*batch_size + 1;

    dim3 numBlocks_attn(n_blocks_attn_x, n_blocks_attn_y, batch_attn);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(soft_attn_ws.data_ptr<float>(), Values.data_ptr<float>(), num_heads, batch_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, feature_dims, attn.data_ptr<float>());
    cudaDeviceSynchronize();
}