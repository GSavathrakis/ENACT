#ifndef MY_KERNEL_H
#define MY_KERNEL_H

extern "C" __global__ void attention_weights(const float* queries, const float* keys, const int n_heads_bs, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* attn_w);
extern "C" __global__ void softmax(const float* tensor, const int n_heads_bs, const int* start_inds, const int* sizes, const int spat1, const int spat2, float* soft);
extern "C" __global__ void attention(const float* attn_w, const float* val, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_attn_w, const int row_val, const int col_val, float* attn);
extern "C" __global__ void grad_v(const float* soft_attn_ws_tr, const float* grad_outp, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_attn_w_tr, const int row_grad_out, const int col_grad_out, float* grad_val);
extern "C" __global__ void grad_soft_attn_w(const float* grad_outp, const float* values_tr, const int n_heads_bs, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_soft_attn_ws);
extern "C" __global__ void grad_attn_w(const float* grad_soft_attn_ws, const float* soft_attn_ws, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_grad_soft_attn, const int col_grad_soft_attn, float* grad_attn_ws);
extern "C" __global__ void grad_q(const float* grad_attn_ws, const float* keys, const int n_heads_bs, const int* start_inds, const int* sizes, const int row_grad_attn_ws, const int row_keys, const int col_keys, float* grad_queries);
extern "C" __global__ void grad_k(const float* grad_attn_ws, const float* queries, const int n_heads_bs, const int* start_inds, const int* sizes, const int col_grad_attn_ws, const int col_queries, const int row_queries, float* grad_keys);

#endif