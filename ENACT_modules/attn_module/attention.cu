#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

using namespace std;

vector<torch::Tensor> forward_mhsa(const torch::Tensor Queries, const torch::Tensor Keys, const torch::Tensor Values, const torch::Tensor clust_start_inds, 
                  const torch::Tensor clust_sizes){
    // Queries shape: num heads x Batch size x spatial dimensions x feature dimensions
    // Keys shape:    concatenated spatial dims along batch size and num heads x feature dimensions
    // Values shape:  concatenated spatial dims along batch size and num heads x feature dimensions

    torch::Tensor attn_ws = torch::zeros({Queries.size(1), Keys.size(0)}, Queries.options());
    torch::Tensor soft_attn_ws = torch::zeros({Queries.size(1), Keys.size(0)}, Queries.options());

    int nh_bs = Queries.size(0);
    int spatial_size = Queries.size(1);
    int feature_dims = Queries.size(2);
    int concat_spatial_dims = Keys.size(0);

    int n_threads_attn_ws_x = 32;
    int n_threads_attn_ws_y = 32;

    int n_blocks_attn_ws_x = (spatial_size + n_threads_attn_ws_x - 1)/n_threads_attn_ws_x + 1;
    int n_blocks_attn_ws_y = (concat_spatial_dims + n_threads_attn_ws_y - 1)/n_threads_attn_ws_y + 1;
    int batch_attn_ws = nh_bs + 1;

    /*int* clust_start_inds_gpu;
    int* clust_sizes_gpu;

    cudaMalloc(&clust_start_inds_gpu, clust_start_inds.size() * sizeof(int));
    cudaMalloc(&clust_sizes_gpu, clust_sizes.size() * sizeof(int));

    cudaMemcpy(clust_start_inds_gpu, clust_start_inds.data(), clust_start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clust_sizes_gpu, clust_sizes.data(), clust_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);*/

    dim3 numBlocks_attn_ws(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn_ws(n_threads_attn_ws_x, n_threads_attn_ws_y);
    attention_weights<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(Queries.data_ptr<float>(), Keys.data_ptr<float>(), nh_bs, spatial_size, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
    attn_ws = attn_ws/sqrt(feature_dims);

    int n_threads_soft_attn_ws = 1024;

    int n_blocks_soft_attn_ws_x = (spatial_size + n_threads_soft_attn_ws - 1)/n_threads_soft_attn_ws + 1;
    int n_blocks_soft_attn_ws_y = nh_bs + 1;
    
    dim3 numBlocks_soft_attn_ws(n_blocks_soft_attn_ws_x, n_blocks_soft_attn_ws_y);
    dim3 threadsPerBlock_soft_attn_ws(n_threads_soft_attn_ws, 1);
    softmax<<<numBlocks_soft_attn_ws, threadsPerBlock_soft_attn_ws>>>(attn_ws.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_size, concat_spatial_dims, soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    torch::Tensor attn = torch::zeros({Queries.size(0), soft_attn_ws.size(0), Values.size(1)}, Queries.options());

    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (spatial_size + n_threads_attn_x - 1)/n_threads_attn_x + 1;
    int n_blocks_attn_y = (feature_dims + n_threads_attn_y - 1)/n_threads_attn_y + 1;
    int batch_attn = nh_bs + 1;

    dim3 numBlocks_attn(n_blocks_attn_x, n_blocks_attn_y, batch_attn);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(soft_attn_ws.data_ptr<float>(), Values.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_size, concat_spatial_dims, feature_dims, attn.data_ptr<float>());
    cudaDeviceSynchronize();
    //attn = attn.reshape({Queries.size(0), Queries.size(1), soft_attn_ws.size(0), Values.size(1)});

    /*cudaFree(clust_start_inds_gpu);
    cudaFree(clust_sizes_gpu);*/

    return{
        attn, soft_attn_ws
    };

}

vector<torch::Tensor> backward_mhsa(const torch::Tensor grad_output, const torch::Tensor soft_attn_ws, const torch::Tensor Queries, const torch::Tensor Keys, 
                                    const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes){
    
    int nh_bs = grad_output.size(0);
    int spatial_dims = grad_output.size(1);
    int feature_dims = grad_output.size(2);
    int concat_spatial_dims = Keys.size(0);

    torch::Tensor grad_values = torch::zeros({Values.size(0), Values.size(1)}, Values.options());

    int n_threads_grad_v_x = 32;
    int n_threads_grad_v_y = 32;

    int n_blocks_grad_v_x = (concat_spatial_dims + n_threads_grad_v_x - 1)/n_threads_grad_v_x + 1;
    int n_blocks_grad_v_y = (feature_dims + n_threads_grad_v_y - 1)/n_threads_grad_v_y + 1;
    int batch_grad_v = nh_bs + 1;

    dim3 numBlocks_grad_v(n_blocks_grad_v_x, n_blocks_grad_v_y, batch_grad_v);
    dim3 threadsPerBlock_grad_v(n_threads_grad_v_x, n_threads_grad_v_y);
    grad_v<<<numBlocks_grad_v, threadsPerBlock_grad_v>>>(soft_attn_ws.data_ptr<float>(), grad_output.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, spatial_dims, feature_dims, grad_values.data_ptr<float>());
    cudaDeviceSynchronize();

    torch::Tensor grad_soft_attn_ws = torch::zeros({soft_attn_ws.size(0), soft_attn_ws.size(1)}, soft_attn_ws.options());

    int n_threads_grad_soft_attn_w_x = 32;
    int n_threads_grad_soft_attn_w_y = 32;

    int n_blocks_grad_soft_attn_w_x = (spatial_dims + n_threads_grad_soft_attn_w_x - 1)/n_threads_grad_soft_attn_w_x + 1;
    int n_blocks_grad_soft_attn_w_y = (concat_spatial_dims + n_threads_grad_soft_attn_w_y - 1)/n_threads_grad_soft_attn_w_y + 1;
    int batch_grad_soft_attn_w = nh_bs + 1;

    dim3 numBlocks_grad_soft_attn_w(n_blocks_grad_soft_attn_w_x, n_blocks_grad_soft_attn_w_y, batch_grad_soft_attn_w);
    dim3 threadsPerBlock_grad_soft_attn_w(n_threads_grad_soft_attn_w_x, n_threads_grad_soft_attn_w_y);
    grad_soft_attn_w<<<numBlocks_grad_soft_attn_w, threadsPerBlock_grad_soft_attn_w>>>(grad_output.data_ptr<float>(), Values.data_ptr<float>(), nh_bs, spatial_dims, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, grad_soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    torch::Tensor grad_attn_ws = torch::zeros({soft_attn_ws.size(0), soft_attn_ws.size(1)}, soft_attn_ws.options());

    int n_threads_grad_attn_w_x = 32;
    int n_threads_grad_attn_w_y = 32;

    int n_blocks_grad_attn_w_x = (concat_spatial_dims + n_threads_grad_attn_w_x - 1)/n_threads_grad_attn_w_x + 1;
    int n_blocks_grad_attn_w_y = (spatial_dims + n_threads_grad_attn_w_y - 1)/n_threads_grad_attn_w_y + 1;
    int batch_grad_attn_w = nh_bs + 1;

    dim3 numBlocks_grad_attn_w(n_blocks_grad_attn_w_x, n_blocks_grad_attn_w_y, batch_grad_attn_w);
    dim3 threadsPerBlock_grad_attn_w(n_threads_grad_attn_w_x, n_threads_grad_attn_w_y);
    grad_attn_w<<<numBlocks_grad_attn_w, threadsPerBlock_grad_attn_w>>>(grad_soft_attn_ws.data_ptr<float>(), soft_attn_ws.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, grad_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    torch::Tensor grad_queries = torch::zeros({Queries.size(0),Queries.size(1),Queries.size(2)}, Queries.options());

    int n_threads_grad_q_x = 32;
    int n_threads_grad_q_y = 32;

    int n_blocks_grad_q_x = (spatial_dims + n_threads_grad_q_x - 1)/n_threads_grad_q_x + 1;
    int n_blocks_grad_q_y = (feature_dims + n_threads_grad_q_y - 1)/n_threads_grad_q_y + 1;
    int batch_grad_q = nh_bs + 1;

    dim3 numBlocks_grad_q(n_blocks_grad_q_x, n_blocks_grad_q_y, batch_grad_q);
    dim3 threadsPerBlock_grad_q(n_threads_grad_q_x, n_threads_grad_q_y);
    grad_q<<<numBlocks_grad_q, threadsPerBlock_grad_q>>>(grad_attn_ws.data_ptr<float>(), Keys.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), spatial_dims, concat_spatial_dims, feature_dims, grad_queries.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_queries = grad_queries/sqrt(feature_dims);

    torch::Tensor grad_keys = torch::zeros({Keys.size(0), Keys.size(1)}, Keys.options());

    int n_threads_grad_k_x = 32;
    int n_threads_grad_k_y = 32;

    int n_blocks_grad_k_x = (concat_spatial_dims + n_threads_grad_k_x - 1)/n_threads_grad_k_x + 1;
    int n_blocks_grad_k_y = (feature_dims + n_threads_grad_k_y - 1)/n_threads_grad_k_y + 1;
    int batch_grad_k = nh_bs + 1;

    dim3 numBlocks_grad_k(n_blocks_grad_k_x, n_blocks_grad_k_y, batch_grad_k);
    dim3 threadsPerBlock_grad_k(n_threads_grad_k_x, n_threads_grad_k_y);
    grad_k<<<numBlocks_grad_k, threadsPerBlock_grad_k>>>(grad_attn_ws.data_ptr<float>(), Queries.data_ptr<float>(), nh_bs, clust_start_inds.data_ptr<int>(), clust_sizes.data_ptr<int>(), concat_spatial_dims, feature_dims, spatial_dims, grad_keys.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_keys = grad_keys/sqrt(feature_dims);

    return {
        grad_queries, grad_keys, grad_values
    };
}