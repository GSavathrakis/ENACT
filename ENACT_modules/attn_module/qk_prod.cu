#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void qk_product(const torch::Tensor Queries, const torch::Tensor Keys, const torch::Tensor clust_start_inds, 
                const torch::Tensor clust_sizes, const int num_heads, const int batch_size, const int spatial_size, 
                const int feature_dims, const int concat_spatial_dims, torch::Tensor attn_ws){

    int n_threads_attn_ws_x = 32;
    int n_threads_attn_ws_y = 32;

    int n_blocks_attn_ws_x = (spatial_size + n_threads_attn_ws_x - 1)/n_threads_attn_ws_x + 1;
    int n_blocks_attn_ws_y = (concat_spatial_dims + n_threads_attn_ws_y - 1)/n_threads_attn_ws_y + 1;
    int batch_attn_ws = num_heads*batch_size + 1;

    dim3 numBlocks_attn_ws(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn_ws(n_threads_attn_ws_x, n_threads_attn_ws_y);
    attention_weights<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(Queries.data_ptr<float>(), Keys.data_ptr<float>(), num_heads, batch_size, spatial_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, attn_ws.data_ptr<float>());
    attn_ws = attn_ws/sqrt(feature_dims);
    cudaDeviceSynchronize();
}