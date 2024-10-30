#include "ops.h"
#include <cmath>

__global__ void attention_weights(const float* queries, const float* keys, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* attn_w){
    
    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;
    int id2 = blockIdx.y*blockDim.y + threadIdx.y;

    if (bs_n_heads<batch_size*n_heads && id1<spat1 && id2 >= start_inds[bs_n_heads] && id2 < start_inds[bs_n_heads] + sizes[bs_n_heads]){
        float sum=0.;
        for (int d=0; d<feature_dims; d++){
            sum += queries[bs_n_heads*spat1*feature_dims + id1*feature_dims + d]*keys[id2*feature_dims + d];
        }
        attn_w[id1*spat2 + id2] = sum;
    }
}

__global__ void softmax(const float* tensor, const int n_heads, const int batch_size, const int* start_inds, const int* sizes, const int spat1, const int spat2, float* soft){

    int bs_n_heads = blockIdx.x;
    int id1 = blockIdx.y*blockDim.y + threadIdx.y;

    if (id1 < spat1 && bs_n_heads<batch_size*n_heads){
        float sum=0.;
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            sum+=exp(tensor[id1*spat2+s]);
        }
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            soft[id1*spat2+s] = exp(tensor[id1*spat2+s])/sum;
        }
    }
}

__global__ void attention(const float* attn_w, const float* values, const int n_heads, const int batch_size, const int* start_inds, const int* sizes, const int spat1, const int spat2, const int feature_dims, float* attn){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x; // Concerns the attn_ws
    int id2 = blockIdx.y*blockDim.y + threadIdx.y; // Concerns the Values

    if (bs_n_heads<batch_size*n_heads && id1<spat1 && id2<feature_dims){
        float sum=0.;
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads] + sizes[bs_n_heads]; s++){
            sum+=attn_w[id1*spat2+s]*values[id2*spat2+s];
        }
        attn[bs_n_heads*spat1*feature_dims + id1*feature_dims + id2] = sum;
    }
}

__global__ void grad_v(const float* grad_outp, const float* soft_attn_w_tr, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_val){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;
    int id2 = blockIdx.y*blockDim.y + threadIdx.y;

    if (bs_n_heads<batch_size*n_heads && id1>=start_inds[bs_n_heads] && id1<start_inds[bs_n_heads] + sizes[bs_n_heads] && id2<feature_dims){
        float sum=0.;
        for (int s=0;s<spat1;s++){
            sum+=soft_attn_w_tr[id1*spat1+s]*grad_outp[bs_n_heads*feature_dims*spat1 + id2*spat1 + s];
        }
        grad_val[id1*feature_dims + id2] = sum;
    }
}

__global__ void grad_soft_attn_w(const float* grad_outp, const float* values_tr, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_soft_attn_ws){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x; // Concerns the grad output
    int id2 = blockIdx.y*blockDim.y + threadIdx.y; // Concerns the values_tr

    if (bs_n_heads<batch_size*n_heads && id1<spat1 && id2>=start_inds[bs_n_heads] && id2<start_inds[bs_n_heads] + sizes[bs_n_heads]){
        float sum=0.;
        for (int d=0;d<feature_dims;d++){
            sum+=grad_outp[bs_n_heads*spat1*feature_dims + id1*feature_dims + d]*values_tr[id2*feature_dims + d];
        }
        grad_soft_attn_ws[id1*spat2 + id2] = sum;
    }
}

__global__ void grad_attn_w(const float* grad_soft_attn_ws, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, float* grad_attn_ws){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;
    int id2 = blockIdx.y*blockDim.y + threadIdx.y;

    if (bs_n_heads<batch_size*n_heads && id1<spat1 && id2>=start_inds[bs_n_heads] && id2<start_inds[bs_n_heads]+sizes[bs_n_heads]){
        float sum=0.;
        for (int n=start_inds[bs_n_heads]; n<start_inds[bs_n_heads]+sizes[bs_n_heads]; n++){
            if (n==id2){
                sum+=grad_soft_attn_ws[id1*spat2+id2]*(1-grad_soft_attn_ws[id1*spat2+n]);
            }
            else{
                sum+=-grad_soft_attn_ws[id1*spat2+id2]*grad_soft_attn_ws[id1*spat2+n];
            }
        }
        grad_attn_ws[id1*spat2+id2]=sum;
        /*for (int m=start_inds[bs_n_heads]; m<start_inds[bs_n_heads]+sizes[bs_n_heads]; m++){
            float sum=0.;
            for (int n=start_inds[bs_n_heads]; n<start_inds[bs_n_heads]+sizes[bs_n_heads]; n++){
                if (m==n){
                    sum+=grad_soft_attn_ws[id1*spat2+m]*(1-grad_soft_attn_ws[id1*spat2+n]);
                }
                else{
                    sum+=-grad_soft_attn_ws[id1*spat2+m]*grad_soft_attn_ws[id1*spat2+n];
                }
                __syncthreads();
            }
            grad_attn_ws[id1*spat2+m]=sum;
        }*/
    }

}

__global__ void grad_q(const float* grad_attn_ws, const float* keys, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_queries){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x; // Concerns the grad attn ws
    int id2 = blockIdx.y*blockDim.y + threadIdx.y; // Concerns the keys

    if (bs_n_heads<batch_size*n_heads && id1<spat1 && id2<feature_dims){
        float sum=0.;
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            sum+=grad_attn_ws[id1*spat2+s]*keys[id2*spat2+s];
        }
        grad_queries[bs_n_heads*spat1*feature_dims + id1*feature_dims + id2] = sum;
    }
}

__global__ void grad_k(const float* grad_attn_ws_tr, const float* queries, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_keys){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x; // Concerns the grad attn ws transposed
    int id2 = blockIdx.y*blockDim.y + threadIdx.y; // Concerns the queries

    if (bs_n_heads<batch_size*n_heads && id1>=start_inds[bs_n_heads] && id1<start_inds[bs_n_heads] + sizes[bs_n_heads] && id2<feature_dims){
        float sum=0.;
        for (int s=0;s<spat1;s++){
            sum+=grad_attn_ws_tr[id1*spat1 + s]*queries[bs_n_heads*feature_dims*spat1 + id2*spat1 + s];
        }
        grad_keys[id1*feature_dims+id2] = sum;
    }
}