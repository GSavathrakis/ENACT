#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "attention.h"
#include "ops/ops.h"
#include <chrono>

using namespace std;

vector<at::Tensor> forward_mhsa(at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes){
    // Queries shape: num heads x Batch size x spatial dimensions x feature dimensions
    // Keys shape:    concatenated spatial dims along batch size and num heads x feature dimensions
    // Values shape:  concatenated spatial dims along batch size and num heads x feature dimensions

    at::Tensor attn_ws = at::zeros({Queries.size(2), Keys.size(0)}, Queries.options());
    at::Tensor soft_attn_ws = at::zeros({Queries.size(2), Keys.size(0)}, Queries.options());

    int n_threads_attn_ws_x = 32;
    int n_threads_attn_ws_y = 32;

    int n_blocks_attn_ws_x = (Queries.size(2) + n_threads_attn_ws_x - 1)/n_threads_attn_ws_x;
    int n_blocks_attn_ws_y = (Keys.size(0) + n_threads_attn_ws_y - 1)/n_threads_attn_ws_y;
    int batch_attn_ws = Queries.size(0)*Queries.size(1);

    int* clust_start_inds_gpu;
    int* clust_sizes_gpu;

    cudaMalloc(&clust_start_inds_gpu, clust_start_inds.size() * sizeof(int));
    cudaMalloc(&clust_sizes_gpu, clust_sizes.size() * sizeof(int));

    cudaMemcpy(clust_start_inds_gpu, clust_start_inds.data(), clust_start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clust_sizes_gpu, clust_sizes.data(), clust_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks_attn_ws(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn_ws(n_threads_attn_ws_x, n_threads_attn_ws_y);
    attention_weights<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(Queries.data_ptr<float>(), Keys.data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Keys.size(0), Queries.size(3), attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
    attn_ws = attn_ws/sqrt(Keys.size(1));

    int n_threads_soft_attn_ws = 1024;

    int n_blocks_soft_attn_ws_x = Queries.size(0)*Queries.size(1);
    int n_blocks_soft_attn_ws_y = (Queries.size(2) + n_threads_soft_attn_ws - 1)/n_threads_soft_attn_ws;
    
    dim3 numBlocks_soft_attn_ws(n_blocks_soft_attn_ws_x, n_blocks_soft_attn_ws_y);
    dim3 threadsPerBlock_soft_attn_ws(1, n_threads_soft_attn_ws);
    softmax<<<numBlocks_soft_attn_ws, threadsPerBlock_soft_attn_ws>>>(attn_ws.data_ptr<float>(), Queries.size(0), Queries.size(1), clust_start_inds_gpu, clust_sizes_gpu, Queries.size(2), Keys.size(0), soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    at::Tensor attn = at::zeros({Queries.size(0)*Queries.size(1), Queries.size(2), Queries.size(3)}, Queries.options());

    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (soft_attn_ws.size(0) + n_threads_attn_x - 1)/n_threads_attn_x;
    int n_blocks_attn_y = (Values.size(1) + n_threads_attn_y - 1)/n_threads_attn_y;
    int batch_attn = Queries.size(0)*Queries.size(1);
    // The problem is HERE at computing ATTENTION
    dim3 numBlocks_attn(n_blocks_attn_x, n_blocks_attn_y, batch_attn);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(soft_attn_ws.data_ptr<float>(), Values.transpose(0,1).data_ptr<float>(), Queries.size(0), Queries.size(1), clust_start_inds_gpu, clust_sizes_gpu, Queries.size(2), Keys.size(0), Queries.size(3), attn.data_ptr<float>());
    cudaDeviceSynchronize();
    attn = attn.reshape({Queries.size(0), Queries.size(1), Queries.size(2), Queries.size(3)});

    cudaFree(clust_start_inds_gpu);
    cudaFree(clust_sizes_gpu);

    return{
        attn, soft_attn_ws
    };

}

vector<at::Tensor> backward_mhsa(at::Tensor grad_output, at::Tensor soft_attn_ws, at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes){
    
    at::Tensor grad_values = at::zeros({Values.size(0), Values.size(1)}, Values.options());
    grad_output = grad_output.reshape({grad_output.size(0)*grad_output.size(1), grad_output.size(2), grad_output.size(3)});

    int n_threads_grad_v_x = 32;
    int n_threads_grad_v_y = 32;

    int n_blocks_grad_v_x = (soft_attn_ws.size(1) + n_threads_grad_v_x - 1)/n_threads_grad_v_x;
    int n_blocks_grad_v_y = (grad_output.size(1) + n_threads_grad_v_y - 1)/n_threads_grad_v_y;
    int batch_grad_v = grad_output.size(0);

    int* clust_start_inds_gpu;
    int* clust_sizes_gpu;

    cudaMalloc(&clust_start_inds_gpu, clust_start_inds.size() * sizeof(int));
    cudaMalloc(&clust_sizes_gpu, clust_sizes.size() * sizeof(int));

    cudaMemcpy(clust_start_inds_gpu, clust_start_inds.data(), clust_start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clust_sizes_gpu, clust_sizes.data(), clust_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks_grad_v(n_blocks_grad_v_x, n_blocks_grad_v_y, batch_grad_v);
    dim3 threadsPerBlock_grad_v(n_threads_grad_v_x, n_threads_grad_v_y);
    grad_v<<<numBlocks_grad_v, threadsPerBlock_grad_v>>>(grad_output.transpose(1,2).data_ptr<float>(), soft_attn_ws.transpose(0,1).data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Values.size(0), Queries.size(3), grad_values.data_ptr<float>());
    cudaDeviceSynchronize();

    at::Tensor grad_soft_attn_ws = at::zeros({grad_output.size(1), Values.size(0)}, grad_output.options());
    
    int n_threads_grad_soft_attn_w_x = 32;
    int n_threads_grad_soft_attn_w_y = 32;

    int n_blocks_grad_soft_attn_w_x = (grad_output.size(1) + n_threads_grad_soft_attn_w_x - 1)/n_threads_grad_soft_attn_w_x;
    int n_blocks_grad_soft_attn_w_y = (Values.size(0) + n_threads_grad_soft_attn_w_y - 1)/n_threads_grad_soft_attn_w_y;
    int batch_grad_soft_attn_w = grad_output.size(0);

    dim3 numBlocks_grad_soft_attn_w(n_blocks_grad_soft_attn_w_x, n_blocks_grad_soft_attn_w_y, batch_grad_soft_attn_w);
    dim3 threadsPerBlock_grad_soft_attn_w(n_threads_grad_soft_attn_w_x, n_threads_grad_soft_attn_w_y);
    grad_soft_attn_w<<<numBlocks_grad_soft_attn_w, threadsPerBlock_grad_soft_attn_w>>>(grad_output.data_ptr<float>(), Values.data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Values.size(0), Queries.size(3), grad_soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    at::Tensor grad_attn_ws = at::zeros({grad_output.size(1), Values.size(0)}, grad_output.options());

    int n_threads_grad_attn_w_x = 32;
    int n_threads_grad_attn_w_y = 32;

    int n_blocks_grad_attn_w_x = (grad_output.size(1) + n_threads_grad_attn_w_x - 1)/n_threads_grad_attn_w_x;
    int n_blocks_grad_attn_w_y = (Values.size(0) + n_threads_grad_attn_w_y - 1)/n_threads_grad_attn_w_y;
    int batch_grad_attn_w = grad_output.size(0);

    dim3 numBlocks_grad_attn_w(n_blocks_grad_attn_w_x, n_blocks_grad_attn_w_y, batch_grad_attn_w);
    dim3 threadsPerBlock_grad_attn_w(n_threads_grad_attn_w_x, n_threads_grad_attn_w_y);
    grad_attn_w<<<numBlocks_grad_attn_w, threadsPerBlock_grad_attn_w>>>(grad_soft_attn_ws.data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Values.size(0), grad_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
    //delete grad_soft_attn_ws;

    at::Tensor grad_queries = at::zeros({grad_output.size(0), grad_output.size(1), grad_output.size(2)}, grad_output.options());

    int n_threads_grad_q_x = 32;
    int n_threads_grad_q_y = 32;

    int n_blocks_grad_q_x = (grad_attn_ws.size(0) + n_threads_grad_q_x - 1)/n_threads_grad_q_x;
    int n_blocks_grad_q_y = (Keys.size(1) + n_threads_grad_q_y - 1)/n_threads_grad_q_y;
    int batch_grad_q = grad_output.size(0);

    dim3 numBlocks_grad_q(n_blocks_grad_q_x, n_blocks_grad_q_y, batch_grad_q);
    dim3 threadsPerBlock_grad_q(n_threads_grad_q_x, n_threads_grad_q_y);
    grad_q<<<numBlocks_grad_q, threadsPerBlock_grad_q>>>(grad_attn_ws.data_ptr<float>(), Keys.transpose(0,1).data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Values.size(0), Queries.size(3), grad_queries.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_queries = grad_queries.reshape({Queries.size(0), Queries.size(1), Queries.size(2), Queries.size(3)});

    at::Tensor grad_keys = at::zeros({Keys.size(0), Keys.size(1)}, Keys.options());

    int n_threads_grad_k_x = 32;
    int n_threads_grad_k_y = 32;

    int n_blocks_grad_k_x = (grad_attn_ws.size(1) + n_threads_grad_k_x - 1)/n_threads_grad_k_x;
    int n_blocks_grad_k_y = (Queries.size(2) + n_threads_grad_k_y - 1)/n_threads_grad_k_y;
    int batch_grad_k = grad_output.size(0);

    dim3 numBlocks_grad_k(n_blocks_grad_k_x, n_blocks_grad_k_y, batch_grad_k);
    dim3 threadsPerBlock_grad_k(n_threads_grad_k_x, n_threads_grad_k_y);
    grad_k<<<numBlocks_grad_k, threadsPerBlock_grad_k>>>(grad_attn_ws.transpose(0,1).data_ptr<float>(), Queries.transpose(1,2).data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), clust_start_inds_gpu, clust_sizes_gpu, Values.size(0), Queries.size(3), grad_keys.data_ptr<float>());
    cudaDeviceSynchronize();
    //delete grad_attn_ws;

    cudaFree(clust_start_inds_gpu);
    cudaFree(clust_sizes_gpu);

    return{
        grad_queries, grad_keys, grad_values
    };

}