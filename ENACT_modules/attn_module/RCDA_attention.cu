#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

using namespace std;

torch::Tensor forward_rcda_w(const torch::Tensor Q, const torch::Tensor K, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes){
    
    // Q shape: num heads times batch size x row or column x embedding dimensions
    // K shape: concatenated spatial dims along batch size and num heads x embedding dimensions

    torch::Tensor attn_ws = torch::zeros({Q.size(1), K.size(0)}, Q.options());

    int nh_bs = Q.size(0);
    int spatial_size = Q.size(1);
    int feature_dims = Q.size(2);
    int concat_spatial_dims = K.size(0);

    int n_threads_attn_ws_x = 32;
    int n_threads_attn_ws_y = 32;

    int n_blocks_attn_ws_x = (spatial_size + n_threads_attn_ws_x - 1)/n_threads_attn_ws_x + 1;
    int n_blocks_attn_ws_y = (concat_spatial_dims + n_threads_attn_ws_y - 1)/n_threads_attn_ws_y + 1;
    int batch_attn_ws = nh_bs + 1;

    dim3 numBlocks_attn_ws(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn_ws(n_threads_attn_ws_x, n_threads_attn_ws_y);
    attention_weights<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(Q.data_ptr<float>(), K.data_ptr<float>(), nh_bs, spatial_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, attn_ws.data_ptr<float>());
    attn_ws = attn_ws/sqrt(feature_dims);

    n_blocks_attn_ws_x = (spatial_size + 1024 - 1)/1024 + 1;
    n_blocks_attn_ws_y = nh_bs + 1;
    
    numBlocks_attn_ws = dim3(n_blocks_attn_ws_x, n_blocks_attn_ws_y);
    threadsPerBlock_attn_ws = dim3(1024, 1);
    softmax<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(attn_ws.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_size, concat_spatial_dims);

    return attn_ws;
}

torch::Tensor forward_rcda_map(const torch::Tensor weights, const torch::Tensor V, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int nh_bs){
    
    // weights shape: HW x concatenated spatial dims along batch size and num heads
    // V shape: concatenated spatial dims along batch size and num heads x embedding dimensions

    torch::Tensor attn = torch::zeros({nh_bs, weights.size(0), Values.size(1)}, weights.options());

    int spatial_size = weights.size(0);
    int concat_spatial_dims = weights.size(1);
    int feature_dims = Values.size(1);

    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (spatial_size + n_threads_attn_x - 1)/n_threads_attn_x + 1;
    int n_blocks_attn_y = (feature_dims + n_threads_attn_y - 1)/n_threads_attn_y + 1;
    int batch_attn = nh_bs + 1;

    dim3 numBlocks_attn(n_blocks_attn_x, n_blocks_attn_y, batch_attn);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(attn_ws.data_ptr<float>(), Values.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_size, concat_spatial_dims, feature_dims, attn.data_ptr<float>());

    return attn;
}