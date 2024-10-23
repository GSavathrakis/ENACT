#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "attention.h"
#include <chrono>

using namespace std;

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

vector<at::Tensor> backward_mhsa(at::Tensor grad_attn, at::Tensor attn_w, at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes, int total_cl_size){
    at::Tensor grad_queries = at::zeros({grad_attn.size(0), grad_attn.size(1), grad_attn.size(2), grad_attn.size(3)}, grad_attn.options());
    at::Tensor grad_keys    = at::zeros({total_cl_size, grad_attn.size(3)}, grad_attn.options());
    at::Tensor grad_values    = at::zeros({total_cl_size, grad_attn.size(3)}, grad_attn.options());

    at::Tensor grad_attn_w = at::zeros({grad_attn.size(0)*grad_attn.size(1), grad_attn.size(2), total_cl_size}, grad_attn.options());

    // Calculating the grad of the values
    grad_attn = grad_attn.reshape({grad_attn.size(0)*grad_attn.size(1), grad_attn.size(2), grad_attn.size(3)});

    int n_threads_grad_v_x = 32;
    int n_threads_grad_v_y = 32;

    int n_blocks_grad_v_x = (grad_attn.size(0)*grad_attn.size(2) + n_threads_grad_v_x - 1)/n_threads_grad_v_x;
    int n_blocks_grad_v_y = (attn_w.size(0)*attn_w.size(2) + n_threads_grad_v_y - 1)/n_threads_grad_v_y;

    int batch_grad_v = grad_attn.size(0);

    dim3 numBlocks_grad_v(n_blocks_grad_v_x, n_blocks_grad_v_y, batch_grad_v);
    dim3 threadsPerBlock_grad_v(n_threads_grad_v_x, n_threads_grad_v_y);
    dot_product<<<numBlocks_grad_v, threadsPerBlock_grad_v>>>(attn_w.transpose(1,2).data_ptr<float>(), grad_attn.transpose(1,2).data_ptr<float>(), Queries.size(1), Queries.size(0), attn_w.size(2), grad_attn.size(2), grad_attn.size(1), grad_values.data_ptr<float>());
    cudaDeviceSynchronize();

    grad_values = grad_values.permute(1,0,2).sum(axis=-2);

    // Calculating the grad of the attention weights
    int n_threads_grad_attn_w_x = 32;
    int n_threads_grad_attn_w_y = 32;

    int n_blocks_grad_attn_w_x = (Values.size(0) + n_threads_grad_attn_w_x - 1)/n_threads_grad_attn_w_x;
    int n_blocks_grad_attn_w_y = (grad_attn.size(0)*grad_attn.size(1) + n_threads_grad_attn_w_y - 1)/n_threads_grad_attn_w_y;

    int batch_grad_attn_w = grad_attn.size(0);

    dim3 numBlocks_grad_attn_w(n_blocks_grad_attn_w_x, n_blocks_grad_attn_w_y, batch_grad_attn_w);
    dim3 threadsPerBlock_grad_attn_w(n_threads_grad_attn_w_x, n_threads_grad_attn_w_y);
    dot_product<<<numBlocks_grad_attn_w, threadsPerBlock_grad_attn_w>>>(grad_attn.data_ptr<float>(), Values.data_ptr<float>() , Queries.size(1), Queries.size(0), grad_attn.size(1), Values.size(0), Values.size(1), grad_attn_w.data_ptr<float>());
    cudaDeviceSynchronize();

    // Creating the Jacobian to calculate the softmax grad
    at::Tensor Jacobian = at::zeros({grad_attn.size(0)*grad_attn.size(1), grad_attn.size(2), total_cl_size, total_cl_size}, grad_attn.options());

    int n_threads_softmax = 1024;
    int n_blocks_softmax_x = (grad_attn_w.size(2) + n_threads_softmax - 1)/n_threads_softmax;
    int n_blocks_softmax_y = grad_attn_w.size(0)*grad_attn_w.size(1);

    dim3 numBlocks_softmax(n_blocks_softmax_x, n_blocks_softmax_y);
    dim3 threadsPerBlock_softmax(n_threads_softmax, 1);
    create_Jacobian<<<numBlocks_softmax, threadsPerBlock_softmax>>>(grad_attn_w.data_ptr<float>(), grad_attn.size(0)*grad_attn.size(1)*grad_attn.size(2), total_cl_size, total_cl_size, Jacobian.data_ptr<float>());
    cudaDeviceSynchronize();

    // Calculating the grad of the unweighted attention weights
    at::Tensor unw_grad_attn_w = at::zeros({grad_attn.size(0)*grad_attn.size(1), grad_attn.size(2), 1, total_cl_size}, grad_attn.options());

    int n_threads_unw_grad_attn_w_x = 32;
    int n_threads_unw_grad_attn_w_y = 32;

    int n_blocks_unw_grad_attn_w_x = (grad_attn_w.size(0)*grad_attn_w.size(1) + n_threads_unw_grad_attn_w_x - 1)/n_threads_unw_grad_attn_w_x;
    int n_blocks_unw_grad_attn_w_y = (grad_attn_w.size(0)*grad_attn_w.size(1)*grad_attn_w.size(2) + n_threads_unw_grad_attn_w_y - 1)/n_threads_unw_grad_attn_w_y;
    int batch_unw_grad_attn_w = grad_attn_w.size(0)*grad_attn_w.size(1);

    dim3 numBlocks_unw_grad_attn_w(n_blocks_unw_grad_attn_w_x, n_blocks_unw_grad_attn_w_y, batch_unw_grad_attn_w);
    dim3 threadsPerBlock_unw_grad_attn_w(n_threads_unw_grad_attn_w_x, n_threads_unw_grad_attn_w_y);
    dot_product<<<numBlocks_unw_grad_attn_w, threadsPerBlock_unw_grad_attn_w>>>(Jacobian.data_ptr<float>(), grad_attn_w.data_ptr<float>(), Queries.size(1)*Queries.size(0), grad_attn.size(2), total_cl_size, 1, total_cl_size, unw_grad_attn_w.data_ptr<float>());
    cudaDeviceSynchronize();

    unw_grad_attn_w = unw_grad_attn_w.reshape({grad_attn.size(0)*grad_attn.size(1), grad_attn.size(2), total_cl_size});

    // Calculating the grad of the queries
    int n_threads_grad_q_x = 32;
    int n_threads_grad_q_y = 32;

    int n_blocks_grad_q_x = (Keys.size(1) + n_threads_grad_q_x - 1)/n_threads_grad_q_x;
    int n_blocks_grad_q_y = (Queries.size(0)*Queries.size(1)*Queries.size(2) + n_threads_grad_q_y - 1)/n_threads_grad_q_y;
    int batch_grad_q = Queries.size(0)*Queries.size(1);

    dim3 numBlocks_grad_q(n_blocks_grad_q_x, n_blocks_grad_q_y, batch_grad_q);
    dim3 threadsPerBlock_grad_q(n_threads_grad_q_x, n_threads_grad_q_y);
    dot_product<<<numBlocks_grad_q, threadsPerBlock_grad_q>>>(unw_grad_attn_w.data_ptr<float>(), Keys.transpose(0,1).data_ptr<float>(), Queries.size(1), Queries.size(0), unw_grad_attn_w.size(1), Keys.size(1), Keys.size(0), grad_queries.data_ptr<float>());
    cudaDeviceSynchronize();

    grad_queries = (1./sqrt(Queries.size(3)))*grad_queries;
    grad_queries = grad_queries.reshape({Queries.size(0), Queries.size(1), Queries.size(2), Queries.size(3)});

    // Calculating the grad of the Keys
    int n_threads_grad_k_x = 32;
    int n_threads_grad_k_y = 32;

    int n_blocks_grad_k_x = (Queries.size(0)*Queries.size(1)*total_cl_size + n_threads_grad_k_x - 1)/n_threads_grad_k_x;
    int n_blocks_grad_k_y = (Queries.size(0)*Queries.size(1)*Queries.size(3) + n_threads_grad_k_y - 1)/n_threads_grad_k_y;
    int batch_grad_k = Queries.size(0)*Queries.size(1);

    dim3 numBlocks_grad_k(n_blocks_grad_k_x, n_blocks_grad_k_y, batch_grad_k);
    dim3 threadsPerBlock_grad_k(n_threads_grad_k_x, n_threads_grad_k_y);
    dot_product<<<numBlocks_grad_k, threadsPerBlock_grad_k>>>(unw_grad_attn_w.transpose(1,2).data_ptr<float>(), Queries.transpose(1,2).data_ptr<float>(), Queries.size(1), Queries.size(0), unw_grad_attn_w.size(2), Queries.size(2), Queries.size(1), grad_keys.data_ptr<float>());
    cudaDeviceSynchronize();

    grad_keys = (1./sqrt(Queries.size(3)))*grad_keys;
    grad_keys = grad_keys.transpose(1,0,2).sum(axis=-2);

    return {
        grad_queries, grad_keys, grad_values
    };
}