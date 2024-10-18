#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "attention.h"
#include <chrono>

using namespace std;

__global__ void attention_weights(const float* queries, const float* keys, const int n_heads, const int batch_size, const int spatial_sizes_uncl,
                                  const int* spatial_start_ind_cl, const int* spatial_sizes_cl, const int sum_cl_pixels, const int feat_dims, float* attn_w){
    
    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.y*blockDim.y + threadIdx.y;
    int id2 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1 < spatial_sizes_uncl && id2 >= spatial_start_ind_cl[bs_n_heads] && id2 < spatial_sizes_cl[bs_n_heads] + spatial_start_ind_cl[bs_n_heads] && bs_n_heads < batch_size*n_heads){
        float sum=0.;
        for (int d=0; d<feat_dims; d++){
            sum+=queries[bs_n_heads*spatial_sizes_uncl*feat_dims + id1*feat_dims + d]*keys[id2*feat_dims+d];
        }
        attn_w[bs_n_heads*spatial_sizes_uncl*sum_cl_pixels + id1*sum_cl_pixels + id2] = sum;
    }
}

__global__ void softmax(const float* attn_ws, const int batch_size, const int n_heads, const int spatial_1, const int spatial_2, const int* spatial_start_ind_cl, const int* spatial_sizes_cl, float* soft_attn_w){

    int bs_n_heads = blockIdx.y;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1 < spatial_1 && bs_n_heads<batch_size*n_heads){
        float sum=0.;
        for (int k=spatial_start_ind_cl[bs_n_heads]; k<spatial_start_ind_cl[bs_n_heads]+spatial_sizes_cl[bs_n_heads]; k++){
            sum+=exp(attn_ws[bs_n_heads*spatial_1*spatial_2 + id1*spatial_2 + k]);
        }
        for (int k=spatial_start_ind_cl[bs_n_heads]; k<spatial_start_ind_cl[bs_n_heads]+spatial_sizes_cl[bs_n_heads]; k++){
            soft_attn_w[bs_n_heads*spatial_1*spatial_2 + id1*spatial_2 + k] = exp(attn_ws[bs_n_heads*spatial_1*spatial_2 + id1*spatial_2 + k])/sum;
        }
    }
}

__global__ void attention(const float* attn_w, const float* values, const int n_heads, const int batch_size, const int spatial_1, const int feat_dims, const int spatial_2, float* attn){
    
    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.y*blockDim.y + threadIdx.y;
    int id2 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1 < spatial_1 && id2 < feat_dims && bs_n_heads<batch_size*n_heads){
        float sum=0.;
        for (int k=0; k<spatial_2; k++){
            sum+=attn_w[bs_n_heads*spatial_1*spatial_2 + id1*spatial_2 + k]*values[id2*spatial_2 + k];
        }
        attn[bs_n_heads*spatial_1*feat_dims + id1*feat_dims + id2] = sum;
    }
}

vector<at::Tensor> forward_mhsa(at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes){
    // Queries shape: Batch size x num heads x spatial dimensions x feature dimensions
    // Keys shape:    concatenated spatial dims along batch size and num heads x feature dimensions
    // Values shape: Batch size x num heads x spatial dimensions x feature dimensions

    at::Tensor attn_ws = at::zeros({Queries.size(0)*Queries.size(1), Queries.size(2), Keys.size(0)}, Queries.options());
    at::Tensor soft_attn_ws = at::zeros({Queries.size(0)*Queries.size(1), Queries.size(2), Keys.size(0)}, Queries.options());
    at::Tensor attn = at::zeros({Queries.size(0)*Queries.size(1), Queries.size(2), Queries.size(3)}, Queries.options());

    int n_threads_attn_ws_x = 32;
    int n_threads_attn_ws_y = 32;

    int n_blocks_attn_ws_x = (Keys.size(0) + n_threads_attn_ws_x - 1)/n_threads_attn_ws_x;
    int n_blocks_attn_ws_y = (Queries.size(0)*Queries.size(1)*Queries.size(2) + n_threads_attn_ws_y - 1)/n_threads_attn_ws_y;
    int batch_attn_ws = Queries.size(0)*Queries.size(1);

    int* clust_start_inds_gpu;
    int* clust_sizes_gpu;

    cudaMalloc(&clust_start_inds_gpu, clust_start_inds.size() * sizeof(int));
    cudaMalloc(&clust_sizes_gpu, clust_sizes.size() * sizeof(int));

    cudaMemcpy(clust_start_inds_gpu, clust_start_inds.data(), clust_start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(clust_sizes_gpu, clust_sizes.data(), clust_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks_attn_ws(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn_ws(n_threads_attn_ws_x, n_threads_attn_ws_y);
    attention_weights<<<numBlocks_attn_ws, threadsPerBlock_attn_ws>>>(Queries.data_ptr<float>(), Keys.data_ptr<float>(), Queries.size(1), Queries.size(0), Queries.size(2), 
                                                     clust_start_inds_gpu, clust_sizes_gpu, Keys.size(0), Keys.size(1), attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
    attn_ws = attn_ws/sqrt(Keys.size(1));

    int n_threads_soft_attn_ws_x = 1024;
    int n_threads_soft_attn_ws_y = 1;

    int n_blocks_soft_attn_ws_x = (Queries.size(2) + n_threads_soft_attn_ws_x - 1)/n_threads_soft_attn_ws_x;
    int n_blocks_soft_attn_ws_y = Queries.size(0)*Queries.size(1);
    
    dim3 numBlocks_soft_attn_ws(n_blocks_soft_attn_ws_x, n_blocks_soft_attn_ws_y);
    dim3 threadsPerBlock_soft_attn_ws(n_threads_soft_attn_ws_x, n_threads_soft_attn_ws_y);
    softmax<<<numBlocks_soft_attn_ws, threadsPerBlock_soft_attn_ws>>>(attn_ws.data_ptr<float>(), Queries.size(0), Queries.size(1), Queries.size(2), Keys.size(0), clust_start_inds_gpu, clust_sizes_gpu, soft_attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();

    cudaFree(clust_start_inds_gpu);
    cudaFree(clust_sizes_gpu);

    int n_threads_attn_x = 32;
    int n_threads_attn_y = 32;

    int n_blocks_attn_x = (Keys.size(0) + n_threads_attn_x - 1)/n_threads_attn_x;
    int n_blocks_attn_y = (Queries.size(0)*Queries.size(1)*Queries.size(2) + n_threads_attn_y - 1)/n_threads_attn_y;
    int batch_attn = Queries.size(0)*Queries.size(1);

    dim3 numBlocks_attn(n_blocks_attn_ws_x, n_blocks_attn_ws_y, batch_attn_ws);
    dim3 threadsPerBlock_attn(n_threads_attn_x, n_threads_attn_y);
    attention<<<numBlocks_attn, threadsPerBlock_attn>>>(soft_attn_ws.data_ptr<float>(), Values.transpose(0,1).data_ptr<float>(), Queries.size(1), Queries.size(0), Queries.size(2), Queries.size(3), attn_ws.size(2), attn.data_ptr<float>());
    cudaDeviceSynchronize();
    attn = attn.reshape({Queries.size(0), Queries.size(1), Queries.size(2), Queries.size(3)});


    return{
        attn, soft_attn_ws
    };
}

vector<at::Tensor> backward_mhsa(at::Tensor grad_attn, vector<int> clust_start_inds, vector<int> clust_sizes, int total_cl_size){
    at::Tensor grad_queries = at::zeros({grad_attn.size(0), grad_attn.size(1), grad_attn.size(2), grad_attn.size(3)}, grad_attn.options());
    at::Tensor grad_keys    = at::zeros({total_cl_size, grad_attn.size(3)}, grad_attn.options());
    at::Tensor grad_values    = at::zeros({total_cl_size, grad_attn.size(3)}, grad_attn.options());

    return {
        grad_queries, grad_keys, grad_values
    };
}