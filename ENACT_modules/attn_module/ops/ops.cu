#include "ops.h"

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

__global__ void dot_product(const float* tensor1, const float* tensor2, const int n_heads, const int batch_size, const int spatial_dim1, const int spatial_dim2, const int feat_dims, float* result){
    
    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.y*blockDim.y + threadIdx.y;
    int id2 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1<spatial_dim1 && id2<spatial_dim2 && bs_n_heads<batch_size*n_heads){
        float sum=0;
        for (int d=0; d<feat_dims; d++){
            sum+=tensor1[bs_n_heads*spatial_dim1*feat_dims + id1*feat_dims + d]*tensor2[bs_n_heads*feat_dims*spatial_dim2 + id2*feat_dims + d];
        }
        result[bs_n_heads*spatial_dim1*spatial_dim2 + id1*spatial_dim2 + id2] = sum;
    }
}

__global__ void create_Jacobian(const float* tensor, const int Dim1, const int Dim2, const int Dim3, float* J){
    
    int comm_dims = blockIdx.y;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1<Dim2 && comm_dims<Dim1){
        for (int d=0; d<Dim3; d++){
            if (d==id1){
                J[comm_dims*Dim2*Dim3 + id1*Dim3 + d] = tensor[comm_dims*Dim2+id1]*(1-tensor[comm_dims*Dim2+id1]);
            }
            else{
                J[comm_dims*Dim2*Dim3 + id1*Dim3 + d] = -tensor[comm_dims*Dim2+id1]*tensor[comm_dims*Dim2+d];
            }
        }
    }
}