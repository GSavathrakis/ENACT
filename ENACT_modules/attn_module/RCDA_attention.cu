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

    torch::Tensor attn = torch::zeros({nh_bs, weights.size(0), V.size(1)}, weights.options());

    int spatial_size = weights.size(0);
    int concat_spatial_dims = weights.size(1);
    int feature_dims = V.size(1);

    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (spatial_size + n_threads_attn_x - 1)/n_threads_attn_x + 1;
    int n_blocks_attn_y = (feature_dims + n_threads_attn_y - 1)/n_threads_attn_y + 1;
    int batch_attn = nh_bs + 1;

    dim3 numBlocks_attn(n_blocks_attn_x, n_blocks_attn_y, batch_attn);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(weights.data_ptr<float>(), V.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_size, concat_spatial_dims, feature_dims, attn.data_ptr<float>());

    return attn;
}

vector<torch::Tensor> backward_rcda_map(const torch::Tensor grad_output, const torch::Tensor weights, const torch::Tensor V, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int nh_bs){

    torch::Tensor grad_values = torch::zeros({V.size(0), V.size(1)}, V.options());
    torch::Tensor grad_weights = torch::zeros({weights.size(0), weights.size(1)}, weights.options());

    int spatial_dims = weights.size(0);
    int concat_spatial_dims = weights.size(1);
    int feature_dims = V.size(1);

    int n_threads_grad_v_x = 32;
    int n_threads_grad_v_y = 32;

    int n_blocks_grad_v_x = (concat_spatial_dims + n_threads_grad_v_x - 1)/n_threads_grad_v_x + 1;
    int n_blocks_grad_v_y = (feature_dims + n_threads_grad_v_y - 1)/n_threads_grad_v_y + 1;
    int batch_grad_v = nh_bs + 1;

    dim3 numBlocks_grad_v(n_blocks_grad_v_x, n_blocks_grad_v_y, batch_grad_v);
    dim3 threadsPerBlock_grad_v(n_threads_grad_v_x, n_threads_grad_v_y);
    grad_v<<<numBlocks_grad_v, threadsPerBlock_grad_v>>>(weights.data_ptr<float>(), grad_output.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, spatial_dims, feature_dims, grad_values.data_ptr<float>());

    int n_threads_grad_weights_x = 32;
    int n_threads_grad_weights_y = 32;

    int n_blocks_grad_weights_x = (spatial_dims + n_threads_grad_weights_x - 1)/n_threads_grad_weights_x + 1;
    int n_blocks_grad_weights_y = (concat_spatial_dims + n_threads_grad_weights_y - 1)/n_threads_grad_weights_y + 1;
    int batch_grad_weights = nh_bs + 1;

    dim3 numBlocks_grad_weights(n_blocks_grad_weights_x, n_blocks_grad_weights_y, batch_grad_weights);
    dim3 threadsPerBlock_grad_weights(n_threads_grad_weights_x, n_threads_grad_weights_y);
    grad_soft_attn_w<<<numBlocks_grad_weights, threadsPerBlock_grad_weights>>>(grad_output.data_ptr<float>(), V.data_ptr<float>(), nh_bs, spatial_dims, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, grad_weights.data_ptr<float>());

    return{
        grad_weights, grad_values
    };
}

vector<torch::Tensor> backward_rcda_w(const torch::Tensor grad_w, const torch::Tensor w, const torch::Tensor Q, const torch::Tensor K, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes){

    int nh_bs = Q.size(0);
    int spatial_dims = Q.size(1);
    int feature_dims = Q.size(2);
    int concat_spatial_dims = K.size(0);

    torch::Tensor grad_attn_ws = torch::zeros({grad_w.size(0), grad_w.size(1)}, grad_w.options());

    int n_threads_grad_attn_w_x = 32;
    int n_threads_grad_attn_w_y = 32;

    int n_blocks_grad_attn_w_x = (concat_spatial_dims + n_threads_grad_attn_w_x - 1)/n_threads_grad_attn_w_x + 1;
    int n_blocks_grad_attn_w_y = (spatial_dims + n_threads_grad_attn_w_y - 1)/n_threads_grad_attn_w_y + 1;
    int batch_grad_attn_w = nh_bs + 1;

    dim3 numBlocks_grad_attn_w(n_blocks_grad_attn_w_x, n_blocks_grad_attn_w_y, batch_grad_attn_w);
    dim3 threadsPerBlock_grad_attn_w(n_threads_grad_attn_w_x, n_threads_grad_attn_w_y);
    grad_attn_w<<<numBlocks_grad_attn_w, threadsPerBlock_grad_attn_w>>>(grad_w.data_ptr<float>(), w.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, grad_attn_ws.data_ptr<float>());

    torch::Tensor grad_queries = torch::zeros({Q.size(0),Q.size(1),Q.size(2)}, Q.options());

    int n_threads_grad_q_x = 32;
    int n_threads_grad_q_y = 32;

    int n_blocks_grad_q_x = (spatial_dims + n_threads_grad_q_x - 1)/n_threads_grad_q_x + 1;
    int n_blocks_grad_q_y = (feature_dims + n_threads_grad_q_y - 1)/n_threads_grad_q_y + 1;
    int batch_grad_q = nh_bs + 1;

    dim3 numBlocks_grad_q(n_blocks_grad_q_x, n_blocks_grad_q_y, batch_grad_q);
    dim3 threadsPerBlock_grad_q(n_threads_grad_q_x, n_threads_grad_q_y);
    grad_q<<<numBlocks_grad_q, threadsPerBlock_grad_q>>>(grad_attn_ws.data_ptr<float>(), K.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, feature_dims, grad_queries.data_ptr<float>());

    torch::Tensor grad_keys = torch::zeros({K.size(0), K.size(1)}, K.options());

    int n_threads_grad_k_x = 32;
    int n_threads_grad_k_y = 32;

    int n_blocks_grad_k_x = (concat_spatial_dims + n_threads_grad_k_x - 1)/n_threads_grad_k_x + 1;
    int n_blocks_grad_k_y = (feature_dims + n_threads_grad_k_y - 1)/n_threads_grad_k_y + 1;
    int batch_grad_k = nh_bs + 1;

    dim3 numBlocks_grad_k(n_blocks_grad_k_x, n_blocks_grad_k_y, batch_grad_k);
    dim3 threadsPerBlock_grad_k(n_threads_grad_k_x, n_threads_grad_k_y);
    grad_k<<<numBlocks_grad_k, threadsPerBlock_grad_k>>>(grad_attn_ws.data_ptr<float>(), Q.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, spatial_dims, grad_keys.data_ptr<float>());

    return{
        grad_queries, grad_keys
    };
}