#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "attention.h"
#include <chrono>

using namespace std;

__global__ void softmax(float* attn, const int* group_sizes, const int* group_start_indices, const int numGroups){
    int row = blockIdx.x;

    if (row < numGroups){
        int group_start = group_start_indices[row];
        int group_size = group_sizes[row];
        float sum = 0.;
        for (int i=0; i<group_size; i++){
            int index = group_start + i;
            sum+=exp(attn[index]);
        }
        for (int i=0; i<group_size; i++){
            int index = group_start + i;
            attn[index] = exp(attn[index])/sum;
        }
    }
}

__global__ void softmax_Jacobian(const float* y, float* J, int m, int n){
    int index1 = threadIdx.x + blockIdx.x * blockDim.x;
    int index2 = threadIdx.y + blockIdx.y * blockDim.y;
    int index3 = threadIdx.z + blockIdx.z * blockDim.z;

    if (index3 < m && index2 < n && index1 < n){
        if (index1 == index2){
            J[index3*n*n + index2*n + index1] = y[index3*n + index1]*(1.0f - y[index3*n + index1]);
        }
        else{
            J[index3*n*n + index2*n + index1] = -y[index3*n + index1]*y[index3*n + index2];
        }
    }

}

/*__global__ void grad_val_calc(const float* grad_out, const float* attn_w, const int* group_val_start_ids, const int* group_attn_ids, const int* group_attn_sizes, const int spat, float* grad_v){
__global__ void attention_weights(const float* q, const float* k, const int* end_inds_clust, const int* n_clusters, const int* n_clusters_cumsum, const int n_pixels, const int feat_dim, float* attn){

    // Blocks correspond to pixel Ids namely spatial dimensions flattened by batch_size and number of heads
    
    int index_q = blockIdx.x * blockDim.x + threadIdx.x;
    int index_k = blockIdx.y * blockDim.y + threadIdx.y;

    int ImgIdx = index_k/(n_pixels*feat_dim);

    if (ImgIdx==0){
        int index_attn = (index_k/blockDim.y)*blockDim.y*(end_inds_clust[ImgIdx]+1)+index_q;
        if (index_q/feat_dim <= end_inds_clust[ImgIdx] && threadIdx.x == threadIdx.y){
            atomicAdd(&attn[index_attn], k[index_k]*q[index_q]);
        }
    }
    else{
        int index_attn = ((index_k-ImgIdx*n_pixels*blockDim.y)/blockDim.y)*blockDim.y*n_clusters[ImgIdx] + (index_q-(end_inds_clust[ImgIdx-1]+1)*blockDim.x) + n_clusters_cumsum[ImgIdx-1]*n_pixels*blockDim.x;
        if (index_q/feat_dim <= end_inds_clust[ImgIdx] && index_q/feat_dim > end_inds_clust[ImgIdx-1] && threadIdx.x == threadIdx.y){
            atomicAdd(&attn[index_attn], k[index_k]*q[index_q]);
        }
    }
}

__global__ void attention(const float* attention_w, const float* values, const int* group_attn_ids, const int* attn_start_inds, 
                          const int* group_val_ids, const int* val_start_ids, const int* group_sizes, const int spat, float* attention){
    //End inds clust vals is cumsummed
    int index_attn_w = blockIdx.x * blockDim.x + threadIdx.x;
    int index_values = blockIdx.y * blockDim.y + threadIdx.y;
    int index_attn      = blockIdx.z * blockDim.z + threadIdx.z;

    if (group_attn_ids[index_attn_w]/spat==group_val_ids[blockIdx.y] && index_attn_w==attn_start_inds[index_attn_w] && 
        blockIdx.y==val_start_ids[blockIdx.y] && threadIdx.y == threadIdx.z && index_attn/blockDim.z == group_attn_ids[index_attn_w]){
            
        float sum = 0.;

        for (int i=0; i<group_sizes[blockIdx.y]; i++){
            sum += attention_w[index_attn_w + i]*values[blockIdx.y + i*blockDim.y + threadIdx.y];
        }
        
        attention[index_attn] = sum;
        
    }
    
}

    int index_attn_w = blockIdx.x * blockDim.x + threadIdx.x;
    int index_grad_val = blockIdx.z * blockDim.z + threadIdx.z;

    if (group_attn_ids[index_attn_w]/spat == blockIdx.y/spat && blockIdx.z == group_val_start_ids[index_attn_w] + index_attn_w%group_attn_sizes[index_attn_w] && group_attn_ids[index_attn_w]%spat==0 && threadIdx.y == threadIdx.z){
        
        float sum=0.;

        for (int i=0; i<spat; i++){
            sum+=attn_w[index_attn_w + i*group_attn_sizes[index_attn_w]]*grad_out[blockIdx.y/spat + i*blockDim.y + threadIdx.y];
        }

        grad_v[index_grad_val] = sum;
    }
}

__global__ void Jacobian_grad(const float* grad_attn_w, float* grad_softmax, const int* attn_start_inds, const int* group_attn_sizes){
    int index_grad_attn_w  = blockIdx.x * blockDim.x + threadIdx.x;
    int index_grad_softmax = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_grad_attn_w == index_grad_softmax){
        float sum=0.;
        for (int i=0; i<group_attn_sizes[index_grad_attn_w]; i++){
            sum+=pow(grad_attn_w[index_grad_attn_w-attn_start_inds[index_grad_attn_w]+i],2);
        }
        grad_softmax[index_grad_softmax] = grad_attn_w[index_grad_attn_w]*(grad_attn_w[index_grad_attn_w]-sum);
    }


}


vector<at::Tensor> forward_mhsa(at::Tensor query, at::Tensor key, at::Tensor value, int n_heads, vector<int> cluster_end_inds_cumsum, vector<int> n_clusters, vector<int> n_clusters_cumsum, 
                        vector<int> cluster_end_inds_times_spat, vector<int> cluster_start_inds_times_spat_cumsum, vector<int> cluster_end_inds_times_spat_all_pixs_attn, vector<int> cluster_start_inds_times_spat_all_pixs_attn,
                        vector<int> cluster_end_inds_times_spat_all_pixs_vals, vector<int> cluster_start_inds_times_spat_all_pixs_vals, vector<int> group_sizes){
    
    int max_query_size = *max_element(n_clusters.begin(), n_clusters.end());

    int* cluster_end_inds_cumsum_gpu;
    int* n_clusters_gpu;
    int* n_clusters_cumsum_gpu;
    int* cluster_end_inds_times_spat_gpu;
    int* cluster_start_inds_times_spat_cumsum_gpu;
    int* cluster_end_inds_times_spat_all_pixs_attn_gpu;
    int* cluster_start_inds_times_spat_all_pixs_attn_gpu;
    int* cluster_end_inds_times_spat_all_pixs_vals_gpu;
    int* cluster_start_inds_times_spat_all_pixs_vals_gpu;
    int* group_sizes_gpu;
    

    cudaMalloc(&cluster_end_inds_cumsum_gpu, cluster_end_inds_cumsum.size() * sizeof(int));
    cudaMalloc(&n_clusters_gpu, n_clusters.size() * sizeof(int));
    cudaMalloc(&n_clusters_cumsum_gpu, n_clusters_cumsum.size() * sizeof(int));
    cudaMalloc(&cluster_end_inds_times_spat_gpu, cluster_end_inds_times_spat.size() * sizeof(int));
    cudaMalloc(&cluster_start_inds_times_spat_cumsum_gpu, cluster_start_inds_times_spat_cumsum.size() * sizeof(int));
    cudaMalloc(&cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_end_inds_times_spat_all_pixs_attn.size() * sizeof(int));
    cudaMalloc(&cluster_start_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn.size() * sizeof(int));
    cudaMalloc(&cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_end_inds_times_spat_all_pixs_vals.size() * sizeof(int));
    cudaMalloc(&cluster_start_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals.size() * sizeof(int));
    cudaMalloc(&group_sizes_gpu, group_sizes.size() * sizeof(int));
    

    cudaMemcpy(cluster_end_inds_cumsum_gpu, cluster_end_inds_cumsum.data(), cluster_end_inds_cumsum.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(n_clusters_gpu, n_clusters.data(), n_clusters.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(n_clusters_cumsum_gpu, n_clusters_cumsum.data(), n_clusters_cumsum.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_end_inds_times_spat_gpu, cluster_end_inds_times_spat.data(), cluster_end_inds_times_spat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_start_inds_times_spat_cumsum_gpu, cluster_start_inds_times_spat_cumsum.data(), cluster_start_inds_times_spat_cumsum.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_end_inds_times_spat_all_pixs_attn.data(), cluster_end_inds_times_spat_all_pixs_attn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_start_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn.data(), cluster_start_inds_times_spat_all_pixs_attn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_end_inds_times_spat_all_pixs_vals.data(), cluster_end_inds_times_spat_all_pixs_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_start_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals.data(), cluster_start_inds_times_spat_all_pixs_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_sizes_gpu, group_sizes.data(), group_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);
    

    at::Tensor attn_ws = at::zeros({query.size(0)*query.size(1)*key.size(2), query.size(2)}, query.options());
    printf("Attn_weights\n");
    dim3 numBlocks1(query.size(0)*query.size(1), key.size(0)*key.size(1)*key.size(2));
    dim3 threadsPerBlock1(query.size(2), key.size(3));
    auto beg = chrono::high_resolution_clock::now();
    attention_weights<<<numBlocks1, threadsPerBlock1>>>(query.data_ptr<float>(), key.data_ptr<float>(), cluster_end_inds_cumsum_gpu, n_clusters_gpu, n_clusters_cumsum_gpu, key.size(2), key.size(3), attn_ws.data_ptr<float>());
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - beg);
    cout << "Elapsed Time: " << duration.count();
    attn_ws = attn_ws/sqrt(key.size(3));
    attn_ws = attn_ws.sum(-1);
    
    printf("softmax\n");
    dim3 numBlocks2(n_heads*key.size(1)*key.size(2));
    dim3 threadsPerBlock2(max_query_size);
    softmax<<<numBlocks2, threadsPerBlock2>>>(attn_ws.data_ptr<float>(), cluster_end_inds_times_spat_gpu, cluster_start_inds_times_spat_cumsum_gpu, cluster_end_inds_times_spat.size());
    cudaDeviceSynchronize();
    
    at::Tensor attn = at::zeros({key.size(0),key.size(1),key.size(2),key.size(3)}, key.options());
    printf("Attention\n");
    dim3 numBlocks3(attn_ws.size(0), query.size(0)*query.size(1), n_heads*key.size(1)*key.size(2));
    dim3 threadsPerBlock3(1, key.size(3), key.size(3));
    attention<<<numBlocks3, threadsPerBlock3>>>(attn_ws.data_ptr<float>(), value.data_ptr<float>(), cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn_gpu, cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals_gpu, group_sizes_gpu, key.size(2), attn.data_ptr<float>());
    cudaDeviceSynchronize();
    
    

    cudaFree(cluster_end_inds_cumsum_gpu);
    cudaFree(n_clusters_gpu);
    cudaFree(n_clusters_cumsum_gpu);
    cudaFree(cluster_end_inds_times_spat_gpu);
    cudaFree(cluster_start_inds_times_spat_cumsum_gpu);
    cudaFree(cluster_end_inds_times_spat_all_pixs_attn_gpu);
    cudaFree(cluster_start_inds_times_spat_all_pixs_attn_gpu);
    cudaFree(cluster_end_inds_times_spat_all_pixs_vals_gpu);
    cudaFree(cluster_start_inds_times_spat_all_pixs_vals_gpu);
    cudaFree(group_sizes_gpu);
    

    return {
        attn, attn_ws
    };
    


}

vector<at::Tensor> backward_mhsa(at::Tensor grad_out, at::Tensor attn_w, at::Tensor Value, at::Tensor Key, at::Tensor Query, vector<int> cluster_start_inds_all_pixs_all_inds_vals, vector<int> cluster_end_inds_times_spat_all_pixs_attn, vector<int> group_sizes, int total_spat_cl, 
                                 vector<int> cluster_end_inds_cumsum, vector<int> n_clusters, vector<int> n_clusters_cumsum, vector<int> cluster_start_inds_times_spat_all_pixs_attn, vector<int> group_sizes_all_pixs,
                                 vector<int> cluster_end_inds_times_spat_all_pixs_vals, vector<int> cluster_start_inds_times_spat_all_pixs_vals){
    
    at::Tensor grad_values = at::zeros({grad_out.size(0), total_spat_cl, grad_out.size(3)}, grad_out.options());
    at::Tensor grad_query  = at::zeros({Query.size(0), Query.size(1), Query.size(2)}, Query.options());
    at::Tensor grad_key    = at::zeros({Key.size(0), Key.size(1), Key.size(2), Key.size(3)}, Key.options());
    
    int* cluster_start_inds_all_pixs_all_inds_vals_gpu;
    int* cluster_end_inds_times_spat_all_pixs_attn_gpu;
    int* group_sizes_gpu;

    cudaMalloc(&cluster_start_inds_all_pixs_all_inds_vals_gpu, cluster_start_inds_all_pixs_all_inds_vals.size() * sizeof(int));
    cudaMalloc(&cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_end_inds_times_spat_all_pixs_attn.size() * sizeof(int));
    cudaMalloc(&group_sizes_gpu, group_sizes.size() * sizeof(int));

    cudaMemcpy(cluster_start_inds_all_pixs_all_inds_vals_gpu, cluster_start_inds_all_pixs_all_inds_vals.data(), cluster_start_inds_all_pixs_all_inds_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_end_inds_times_spat_all_pixs_attn.data(), cluster_end_inds_times_spat_all_pixs_attn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_sizes_gpu, group_sizes.data(), group_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks1(attn_w.size(0), grad_out.size(0)*grad_out.size(1)*grad_out.size(2), grad_out.size(0)*total_spat_cl);
    dim3 threadsPerBlock1(1, grad_out.size(3), grad_out.size(3));
    grad_val_calc<<<numBlocks1, threadsPerBlock1>>>(grad_out.data_ptr<float>(), attn_w.data_ptr<float>(), cluster_start_inds_all_pixs_all_inds_vals_gpu, cluster_end_inds_times_spat_all_pixs_attn_gpu, group_sizes_gpu, grad_out.size(2), grad_values.data_ptr<float>());
    cudaDeviceSynchronize();

    

    
    
    
    int* cluster_end_inds_cumsum_gpu;
    int* n_clusters_gpu;
    int* n_clusters_cumsum_gpu;

    cudaMalloc(&cluster_end_inds_cumsum_gpu, cluster_end_inds_cumsum.size() * sizeof(int));
    cudaMalloc(&n_clusters_gpu, n_clusters.size() * sizeof(int));
    cudaMalloc(&n_clusters_cumsum_gpu, n_clusters_cumsum.size() * sizeof(int));

    cudaMemcpy(cluster_end_inds_cumsum_gpu, cluster_end_inds_cumsum.data(), cluster_end_inds_cumsum.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(n_clusters_gpu, n_clusters.data(), n_clusters.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(n_clusters_cumsum_gpu, n_clusters_cumsum.data(), n_clusters_cumsum.size() * sizeof(int), cudaMemcpyHostToDevice);

    at::Tensor grad_attn_w = at::zeros({grad_out.size(0)*total_spat_cl*grad_out.size(2)}, grad_out.options());
    dim3 numBlocks2(Value.size(0)*Value.size(1), grad_out.size(0)*grad_out.size(1)*grad_out.size(2));
    dim3 threadsPerBlock2(Value.size(2), grad_out.size(3));
    attention_weights<<<numBlocks2, threadsPerBlock2>>>(Value.data_ptr<float>(), grad_out.data_ptr<float>(), cluster_end_inds_cumsum_gpu, n_clusters_gpu, n_clusters_cumsum_gpu, grad_out.size(2), grad_out.size(3), grad_attn_w.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_attn_w = grad_attn_w.sum(-1);

    cudaFree(cluster_end_inds_cumsum_gpu);
    cudaFree(n_clusters_gpu);
    cudaFree(n_clusters_cumsum_gpu);
    
    int* cluster_start_inds_times_spat_all_pixs_attn_gpu;
    int* group_sizes_all_pixs_gpu;

    cudaMalloc(&cluster_start_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn.size() * sizeof(int));
    cudaMalloc(&group_sizes_all_pixs_gpu, group_sizes_all_pixs.size() * sizeof(int));

    cudaMemcpy(cluster_start_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn.data(), cluster_start_inds_times_spat_all_pixs_attn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_sizes_all_pixs_gpu, group_sizes_all_pixs.data(), group_sizes_all_pixs.size() * sizeof(int), cudaMemcpyHostToDevice);

    at::Tensor grad_softmax = at::zeros({grad_out.size(0)*total_spat_cl*grad_out.size(2)}, grad_out.options());
    dim3 numBlocks3(grad_out.size(0)*total_spat_cl*grad_out.size(2), grad_out.size(0)*total_spat_cl*grad_out.size(2));
    dim3 threadsPerBlock3(1,1);
    Jacobian_grad<<<numBlocks3, threadsPerBlock3>>>(grad_attn_w.data_ptr<float>(), grad_softmax.data_ptr<float>(), cluster_start_inds_times_spat_all_pixs_attn_gpu, group_sizes_all_pixs_gpu);
    cudaDeviceSynchronize();

    
    cudaFree(group_sizes_all_pixs_gpu);
    
    dim3 numBlocks4(grad_softmax.size(0), Key.size(0)*Key.size(1)*Key.size(2), Query.size(0)*Query.size(1));
    dim3 threadsPerBlock4(1, Key.size(3), Query.size(2));
    grad_val_calc<<<numBlocks4, threadsPerBlock4>>>(Key.data_ptr<float>(), grad_softmax.data_ptr<float>(), cluster_start_inds_all_pixs_all_inds_vals_gpu, cluster_end_inds_times_spat_all_pixs_attn_gpu, group_sizes_gpu, Key.size(2), grad_query.data_ptr<float>());
    cudaDeviceSynchronize();
    
    int* cluster_end_inds_times_spat_all_pixs_vals_gpu;
    int* cluster_start_inds_times_spat_all_pixs_vals_gpu;

    cudaMalloc(&cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_end_inds_times_spat_all_pixs_vals.size() * sizeof(int));
    cudaMalloc(&cluster_start_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals.size() * sizeof(int));

    cudaMemcpy(cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_end_inds_times_spat_all_pixs_vals.data(), cluster_end_inds_times_spat_all_pixs_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cluster_start_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals.data(), cluster_start_inds_times_spat_all_pixs_vals.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks5(grad_softmax.size(0), Query.size(0)*Query.size(1), grad_key.size(0)*grad_key.size(1)*grad_key.size(2));
    dim3 threadsPerBlock5(1, Query.size(2), grad_key.size(3));
    attention<<<numBlocks5, threadsPerBlock5>>>(grad_softmax.data_ptr<float>(), Query.data_ptr<float>(), cluster_end_inds_times_spat_all_pixs_attn_gpu, cluster_start_inds_times_spat_all_pixs_attn_gpu, 
                                                cluster_end_inds_times_spat_all_pixs_vals_gpu, cluster_start_inds_times_spat_all_pixs_vals_gpu, group_sizes_gpu, grad_key.size(2), grad_key.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_query = grad_query/sqrt(Query.size(2));
    grad_key   = grad_key/sqrt(Key.size(3));


    cudaFree(cluster_start_inds_all_pixs_all_inds_vals_gpu);
    cudaFree(cluster_end_inds_times_spat_all_pixs_attn_gpu);
    cudaFree(cluster_start_inds_times_spat_all_pixs_attn_gpu);
    cudaFree(cluster_end_inds_times_spat_all_pixs_vals_gpu);
    cudaFree(cluster_start_inds_times_spat_all_pixs_vals_gpu);
    cudaFree(group_sizes_gpu);

    return {
        grad_query, grad_key, grad_values
    };

}*/





at::Tensor forward_mhsa(at::Tensor query, at::Tensor key, at::Tensor value, int n_heads, list<int> n_clusters){

    //auto query = tens1.to(torch::kCPU);
    //auto key   = tens2.to(torch::kCPU);
    //auto value = tens3.to(torch::kCPU);

    int bs = key.size(1);
    int dk = key.size(3);
    at::Tensor attention = at::zeros({key.size(0),key.size(1),key.size(2),key.size(3)}, key.options());
    for (int m=0; m<n_heads; m++){
        int curr_ind = 0;
        list<int>::iterator it = n_clusters.begin();
        for (int b=0; b<bs; b++){
            advance(it, b);
            int start_ind = curr_ind;
            int end_ind = curr_ind + *it;
            curr_ind = end_ind;
            attention[m][b] = at::matmul(at::softmax(at::matmul(key[m][b], query[m].slice(0, start_ind, end_ind).transpose(0,1))/sqrt(dk), 1), value[m].slice(0, start_ind, end_ind));

        }
    }

    return attention;
}


vector<at::Tensor> backward_mhsa(at::Tensor tens1, at::Tensor tens2, at::Tensor tens3, int n_heads, list<int> n_clusters, at::Tensor grad_output){
    //grads = grads.permute({2,0,1,3});

    //auto grad_output = grads.to(torch::kCPU);

    //auto tens1 = query.to(torch::kCPU);
    //auto tens2 = key.to(torch::kCPU);
    //auto tens3 = value.to(torch::kCPU);

    int bs = grad_output.size(1);
    int dk = grad_output.size(3);


    at::Tensor grad_tens1 = at::zeros({tens1.size(0),tens1.size(1),tens1.size(2)}, tens1.options());
    at::Tensor grad_tens2 = at::zeros({tens2.size(0),tens2.size(1),tens2.size(2),tens2.size(3)}, tens2.options());
    at::Tensor grad_tens3 = at::zeros({tens3.size(0),tens3.size(1),tens3.size(2)}, tens3.options());

    for (int m = 0; m < n_heads; m++){
        int curr_ind = 0;
        list<int>::iterator it = n_clusters.begin();
        for (int b = 0; b < bs; b++){
            advance(it, b);
            int start_ind = curr_ind;
            int end_ind = curr_ind + *it;
            curr_ind = end_ind;
            at::Tensor attn_w = at::softmax(at::matmul(tens2[m][b], tens1[m].slice(0, start_ind, end_ind).transpose(0,1))/sqrt(dk), 1);
            at::Tensor Jacobian = at::zeros({attn_w.size(0), attn_w.size(1), attn_w.size(1)}, attn_w.options());
            grad_tens3[m].slice(0, start_ind, end_ind) = at::matmul(attn_w.transpose(0,1), grad_output[m][b]);
            dim3 numblocks = (attn_w.size(1), attn_w.size(1), attn_w.size(0));
            dim3 threadsperblock = {8, 8, 8};
            softmax_Jacobian<<<numblocks, threadsperblock>>>(attn_w.data_ptr<float>(), Jacobian.data_ptr<float>(), attn_w.size(0), attn_w.size(1));
            //cout << tens2[m][b].transpose(0,1).sizes() << endl;
            grad_tens1[m].slice(0, start_ind, end_ind) = at::matmul((1./sqrt(dk))*tens2[m][b].transpose(0,1), at::matmul(Jacobian, at::matmul(grad_output[m][b], tens3[m].slice(0, start_ind, end_ind).transpose(0,1)).unsqueeze(-1)).squeeze(-1)).transpose(0,1);
            grad_tens2[m][b] = at::matmul(at::matmul(Jacobian, at::matmul(grad_output[m][b], tens3[m].slice(0, start_ind, end_ind).transpose(0,1)).unsqueeze(-1)).squeeze(-1), (1./sqrt(dk))*tens1[m].slice(0, start_ind, end_ind));
        }
    }

    //grad_tens1 = grad_tens1.to(torch::kCUDA);
    //grad_tens2 = grad_tens2.to(torch::kCUDA);
    //grad_tens3 = grad_tens3.to(torch::kCUDA);
    return{
        grad_tens1, grad_tens2, grad_tens3
    };
}