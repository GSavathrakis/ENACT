#include "ops.h"
#include <cmath>

__global__ void attention_weights(const float* queries, const float* keys, const int n_heads_bs, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* attn_w){
    
    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;
    int id2 = blockIdx.y*blockDim.y + threadIdx.y;

    if (bs_n_heads<n_heads_bs && id1<spat1 && id2 >= start_inds[bs_n_heads] && id2 < start_inds[bs_n_heads] + sizes[bs_n_heads]){
        float sum=0.;
        for (int d=0; d<feature_dims; d++){
            sum += queries[bs_n_heads*spat1*feature_dims + id1*feature_dims + d]*keys[id2*feature_dims + d];
        }
        attn_w[id1*spat2 + id2] = sum;
    }
}

__global__ void softmax(const float* tensor, const int n_heads_bs, const int* start_inds, const int* sizes, const int spat1, const int spat2, float* soft){

    int bs_n_heads = blockIdx.y;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x;

    if (id1 < spat1 && bs_n_heads<n_heads_bs){
        float sum=0.;
        float maxx = tensor[id1*spat2+start_inds[bs_n_heads]];
        for (int s=start_inds[bs_n_heads]+1; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            if (tensor[id1*spat2+s]>maxx){
                maxx = tensor[id1*spat2+s];
            }
        }
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            sum+=exp(tensor[id1*spat2+s]-maxx);
        }
        for (int s=start_inds[bs_n_heads]; s<start_inds[bs_n_heads]+sizes[bs_n_heads]; s++){
            soft[id1*spat2+s] = exp(tensor[id1*spat2+s]-maxx)/sum;
        }
    }
}

__global__ void attention(const float* attn_w, const float* val, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_attn_w, const int row_val, const int col_val, float* attn){

    int bs_n_heads = blockIdx.z;
    int id_attn_w = blockIdx.x*blockDim.x + threadIdx.x;
    int id_val = blockIdx.y*blockDim.y+threadIdx.y;

    if (bs_n_heads<n_heads_bs && id_attn_w<row_attn_w && id_val<col_val){
        float sum=0.;
        for (int s=start_inds[bs_n_heads];s<start_inds[bs_n_heads]+sizes[bs_n_heads];s++){
            sum+=attn_w[id_attn_w*row_val+s]*val[s*col_val+id_val];
        }
        attn[bs_n_heads*row_attn_w*col_val+id_attn_w*col_val+id_val]=sum;
    }
}

__global__ void grad_v(const float* soft_attn_ws, const float* grad_outp, const int n_heads_bs, const int* start_inds, const int* sizes, const int col_attn_w, const int row_grad_out, const int col_grad_out, float* grad_val){

    int bs_n_heads  = blockIdx.z;
    int id_attn_ws  = blockIdx.x*blockDim.x+threadIdx.x;
    int id_grad_out = blockIdx.y*blockDim.y+threadIdx.y;

    if (bs_n_heads<n_heads_bs && id_attn_ws>=start_inds[bs_n_heads] && id_attn_ws<start_inds[bs_n_heads]+sizes[bs_n_heads] && id_grad_out<col_grad_out){
        float sum=0.;
        for (int s=0; s<row_grad_out; s++){
            sum+=soft_attn_ws[s*col_attn_w+id_attn_ws]*grad_outp[bs_n_heads*row_grad_out*col_grad_out+s*col_grad_out+id_grad_out];
        }
        grad_val[id_attn_ws*col_grad_out+id_grad_out]=sum;
    }
}

__global__ void grad_soft_attn_w(const float* grad_outp, const float* values_tr, const int n_heads_bs, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_soft_attn_ws){

    int bs_n_heads = blockIdx.z;
    int id1 = blockIdx.x*blockDim.x + threadIdx.x; // Concerns the grad output
    int id2 = blockIdx.y*blockDim.y + threadIdx.y; // Concerns the values_tr

    if (bs_n_heads<n_heads_bs && id1<spat1 && id2>=start_inds[bs_n_heads] && id2<start_inds[bs_n_heads] + sizes[bs_n_heads]){
        float sum=0.;
        for (int d=0;d<feature_dims;d++){
            sum+=grad_outp[bs_n_heads*spat1*feature_dims + id1*feature_dims + d]*values_tr[id2*feature_dims + d];
        }
        grad_soft_attn_ws[id1*spat2 + id2] = sum;
    }
}

__global__ void grad_attn_w(const float* grad_soft_attn_ws, const float* soft_attn_ws, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_grad_soft_attn, const int col_grad_soft_attn, float* grad_attn_ws){

    int bs_n_heads = blockIdx.z;
    int id_col = blockIdx.x*blockDim.x+threadIdx.x;
    int id_row = blockIdx.y*blockDim.y+threadIdx.y;

    if (bs_n_heads<n_heads_bs && id_row<row_grad_soft_attn && id_col>=start_inds[bs_n_heads] && id_col<start_inds[bs_n_heads]+sizes[bs_n_heads]){
        float sum=0.;
        for (int n=start_inds[bs_n_heads];n<start_inds[bs_n_heads]+sizes[bs_n_heads];n++){
            if (n==id_col){
                sum+=soft_attn_ws[id_row*col_grad_soft_attn+id_col]*(1-soft_attn_ws[id_row*col_grad_soft_attn+n])*grad_soft_attn_ws[id_row*col_grad_soft_attn+n];
            }
            else{
                sum-=soft_attn_ws[id_row*col_grad_soft_attn+id_col]*soft_attn_ws[id_row*col_grad_soft_attn+n]*grad_soft_attn_ws[id_row*col_grad_soft_attn+n];
            }
        }
        grad_attn_ws[id_row*col_grad_soft_attn+id_col]=sum;
    }
}

__global__ void grad_q(const float* grad_attn_ws, const float* keys, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_grad_attn_ws, const int row_keys, const int col_keys, float* grad_queries){

    int bs_n_heads = blockIdx.z;
    int id_row_grad_attn = blockIdx.x*blockDim.x+threadIdx.x;
    int id_col_keys = blockIdx.y*blockDim.y+threadIdx.y;

    if (bs_n_heads<n_heads_bs && id_row_grad_attn<row_grad_attn_ws && id_col_keys<col_keys){
        float sum=0.;
        for (int s=start_inds[bs_n_heads];s<start_inds[bs_n_heads]+sizes[bs_n_heads];s++){
            sum+=grad_attn_ws[id_row_grad_attn*row_keys+s]*keys[s*col_keys+id_col_keys];
        }
        grad_queries[bs_n_heads*row_grad_attn_ws*col_keys+id_row_grad_attn*col_keys+id_col_keys]=sum;
    }
}

__global__ void grad_k(const float* grad_attn_ws, const float* queries, const int n_heads_bs, const int* start_inds, const int* sizes, const int col_grad_attn_ws, const int col_queries, const int row_queries, float* grad_keys){

    int bs_n_heads = blockIdx.z;
    int id_col_grad_attn_ws = blockIdx.x*blockDim.x+threadIdx.x;
    int id_col_queries = blockIdx.y*blockDim.y+threadIdx.y;

    if (bs_n_heads<n_heads_bs && id_col_grad_attn_ws>=start_inds[bs_n_heads] && id_col_grad_attn_ws<start_inds[bs_n_heads]+sizes[bs_n_heads] && id_col_queries<col_queries){
        float sum=0.;
        for (int s=0;s<row_queries;s++){
            sum+=grad_attn_ws[s*col_grad_attn_ws+id_col_grad_attn_ws]*queries[bs_n_heads*row_queries*col_queries+s*col_queries+id_col_queries];
        }
        grad_keys[id_col_grad_attn_ws*col_queries+id_col_queries]=sum;
    }
}