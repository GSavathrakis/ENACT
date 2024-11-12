#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "clust_func.h"
using namespace std;

__global__ void clustering(const float* Keys, const float* Values, const float* entropy, const int* entropy_step, const int* start_inds, const int* sizes, const int num_clusters, const int feature_dims, float* Keys_cl, float* Values_cl){

    int regions = blockIdx.x*blockDim.x+threadIdx.x;
    int id_feat = blockIdx.y;

    if (regions<num_clusters && id_feat<feature_dims){
        float sum_k=0.;
        float sum_v=0.;
        float sum_exp=0.;
        for (int s=start_inds[regions];s<start_inds[regions]+sizes[regions];s++){
            if (entropy_step[s]<0){
                sum_k+=exp(entropy[s])*Keys[s*feature_dims+id_feat];
                sum_v+=exp(entropy[s])*Values[s*feature_dims+id_feat];
            }
            else{
                sum_k-=exp(entropy[s])*Keys[s*feature_dims+id_feat];
                sum_v-=exp(entropy[s])*Values[s*feature_dims+id_feat];
            }
            sum_exp+=exp(entropy[s]);
        }
        Keys_cl[regions*feature_dims+id_feat]=sum_k/sum_exp;
        Values_cl[regions*feature_dims+id_feat]=sum_v/sum_exp;
    }
}

__global__ void grad_clustering(const float* grad_Keys_cl, const float* grad_Values_cl, const float* Keys, const float* Values, const float* entropy, const int* entropy_step, const int* start_inds, const int* sizes, const int num_clusters, const int feature_dims, float* grad_Keys, float* grad_Values, float* grad_entropy){

    int regions = blockIdx.x*blockDim.x+threadIdx.x;
    int id_feat = blockIdx.y;

    if (regions<num_clusters && id_feat<feature_dims){
        float sum_exp=0.;
        float sum_k=0.;
        float sum_v=0.;
        for (int s=start_inds[regions];s<start_inds[regions]+sizes[regions];s++){
            sum_exp+=exp(entropy[s]);
            sum_k+=exp(entropy[s])*Keys[s*feature_dims+id_feat];
            sum_v+=exp(entropy[s])*Values[s*feature_dims+id_feat];
        }
        
        for (int s=start_inds[regions];s<start_inds[regions]+sizes[regions];s++){
            if (entropy_step[s]<0){
                grad_Keys[s*feature_dims+id_feat]=grad_Keys_cl[regions*feature_dims+id_feat]*exp(entropy[s])/sum_exp;
                grad_Values[s*feature_dims+id_feat]=grad_Values_cl[regions*feature_dims+id_feat]*exp(entropy[s])/sum_exp;
                grad_entropy[s*feature_dims+id_feat]=-(1./pow(sum_exp,2))*exp(entropy[s])*(grad_Keys_cl[regions*feature_dims+id_feat]*sum_k+grad_Values_cl[regions*feature_dims+id_feat]*sum_v)+(1./sum_exp)*(grad_Keys_cl[regions*feature_dims+id_feat]*exp(entropy[s])*Keys[s*feature_dims+id_feat]+grad_Values_cl[regions*feature_dims+id_feat]*exp(entropy[s])*Values[s*feature_dims+id_feat]);     
            }
            else{
                grad_Keys[s*feature_dims+id_feat]=-grad_Keys_cl[regions*feature_dims+id_feat]*exp(entropy[s])/sum_exp;
                grad_Values[s*feature_dims+id_feat]=-grad_Values_cl[regions*feature_dims+id_feat]*exp(entropy[s])/sum_exp;
                grad_entropy[s*feature_dims+id_feat]=(1./pow(sum_exp,2))*exp(entropy[s])*(grad_Keys_cl[regions*feature_dims+id_feat]*sum_k+grad_Values_cl[regions*feature_dims+id_feat]*sum_v)-(1./sum_exp)*(grad_Keys_cl[regions*feature_dims+id_feat]*exp(entropy[s])*Keys[s*feature_dims+id_feat]+grad_Values_cl[regions*feature_dims+id_feat]*exp(entropy[s])*Values[s*feature_dims+id_feat]);
            }
        }
    }

}


vector<at::Tensor> enact_cluster_forward(at::Tensor Keys, at::Tensor Values, at::Tensor Entropy, vector<int> Entropy_step, vector<int> start_inds, vector<int> region_lengths){
    
    at::Tensor Keys_cl = at::zeros({(int) region_lengths.size(), Keys.size(1)}, Keys.options());
    at::Tensor Values_cl = at::zeros({(int) region_lengths.size(), Values.size(1)}, Values.options());

    int* Entropy_step_gpu;
    int* start_inds_gpu;
    int* region_lengths_gpu;

    cudaMalloc(&Entropy_step_gpu, Entropy_step.size() * sizeof(int));
    cudaMalloc(&start_inds_gpu, start_inds.size() * sizeof(int));
    cudaMalloc(&region_lengths_gpu, region_lengths.size() * sizeof(int));

    cudaMemcpy(Entropy_step_gpu, Entropy_step.data(), Entropy_step.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(start_inds_gpu, start_inds.data(), start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(region_lengths_gpu, region_lengths.data(), region_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    int n_threads_reg = 32;
    int n_threads_ft  = 32;

    int n_blocks_reg = ((int) region_lengths.size() + n_threads_reg - 1)/n_threads_reg;
    int n_blocks_ft = (Keys.size(1) + n_threads_ft - 1)/n_threads_ft;
    //int n_blocks_ft = Keys.size(1);

    dim3 numBlocks(n_blocks_reg, n_blocks_ft);
    dim3 threadsPerBlock(n_threads_reg, n_threads_ft);
    clustering<<<numBlocks, threadsPerBlock>>>(Keys.data_ptr<float>(), Values.data_ptr<float>(), Entropy.data_ptr<float>(), Entropy_step_gpu, start_inds_gpu, region_lengths_gpu, region_lengths.size(), Keys.size(1), Keys_cl.data_ptr<float>(), Values_cl.data_ptr<float>());
    cudaDeviceSynchronize();

    cudaFree(Entropy_step_gpu);
    cudaFree(start_inds_gpu);
    cudaFree(region_lengths_gpu);

    return{
        Keys_cl, Values_cl
    };
}

vector<at::Tensor> enact_cluster_backward(at::Tensor grad_Keys_cl, at::Tensor grad_Values_cl, at::Tensor Keys, at::Tensor Values, at::Tensor Entropy, vector<int> Entropy_step, vector<int> start_inds, vector<int> region_lengths){

    at::Tensor grad_Keys    = at::zeros({Entropy.size(0),   grad_Keys_cl.size(1)},   grad_Keys_cl.options());
    at::Tensor grad_Values  = at::zeros({Entropy.size(0), grad_Values_cl.size(1)}, grad_Values_cl.options());
    at::Tensor grad_entropy = at::zeros({Entropy.size(0), grad_Values_cl.size(1)},        Entropy.options());

    int* Entropy_step_gpu;
    int* start_inds_gpu;
    int* region_lengths_gpu;

    cudaMalloc(&Entropy_step_gpu, Entropy_step.size() * sizeof(int));
    cudaMalloc(&start_inds_gpu, start_inds.size() * sizeof(int));
    cudaMalloc(&region_lengths_gpu, region_lengths.size() * sizeof(int));

    cudaMemcpy(Entropy_step_gpu, Entropy_step.data(), Entropy_step.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(start_inds_gpu, start_inds.data(), start_inds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(region_lengths_gpu, region_lengths.data(), region_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    int n_threads_reg = 32;
    int n_threads_ft  = 32;

    int n_blocks_reg = (grad_Keys_cl.size(0) + n_threads_reg - 1)/n_threads_reg;
    int n_blocks_ft = (grad_Keys_cl.size(1) + n_threads_ft - 1)/n_threads_ft;

    dim3 numBlocks(n_blocks_reg, n_blocks_ft);
    dim3 threadsPerBlock(n_threads_reg);
    grad_clustering<<<numBlocks, threadsPerBlock>>>(grad_Keys_cl.data_ptr<float>(), grad_Values_cl.data_ptr<float>(), Keys.data_ptr<float>(), Values.data_ptr<float>(), Entropy.data_ptr<float>(), Entropy_step_gpu, start_inds_gpu, region_lengths_gpu, region_lengths.size(), Keys.size(1), grad_Keys.data_ptr<float>(), grad_Values.data_ptr<float>(), grad_entropy.data_ptr<float>());
    cudaDeviceSynchronize();
    grad_entropy=grad_entropy.sum(-1);

    cudaFree(Entropy_step_gpu);
    cudaFree(start_inds_gpu);
    cudaFree(region_lengths_gpu);

    return{
        grad_Keys, grad_Values, grad_entropy
    };
}