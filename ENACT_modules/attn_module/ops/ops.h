#ifndef MY_KERNEL_H
#define MY_KERNEL_H

/*extern "C" __global__ void attention_weights(const float* queries, const float* keys, const int n_heads, const int batch_size, const int spatial_sizes_uncl, const int* spatial_start_ind_cl, const int* spatial_sizes_cl, const int sum_cl_pixels, const int feat_dims, float* attn_w);
extern "C" __global__ void softmax(const float* attn_ws, const int batch_size, const int n_heads, const int spatial_1, const int spatial_2, const int* spatial_start_ind_cl, const int* spatial_sizes_cl, float* soft_attn_w);
extern "C" __global__ void attention(const float* attn_w, const float* values, const int n_heads, const int batch_size, const int spatial_1, const int feat_dims, const int spatial_2, float* attn);
extern "C" __global__ void dot_product(const float* tensor1, const float* tensor2, const int n_heads, const int batch_size, const int spatial_dim1, const int spatial_dim2, const int feat_dims, float* result);
extern "C" __global__ void create_Jacobian(const float* tensor, const int Dim1, const int Dim2, const int Dim3, float* J);*/

extern "C" __global__ void attention_weights(const float* queries, const float* keys, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* attn_w);
extern "C" __global__ void softmax(const float* tensor, const int n_heads, const int batch_size, const int* start_inds, const int* sizes, const int spat1, const int spat2, float* soft);
extern "C" __global__ void attention(const float* attn_w, const float* values, const int n_heads, const int batch_size, const int* start_inds, const int* sizes, const int spat1, const int spat2, const int feature_dims, float* attn);
extern "C" __global__ void grad_v(const float* grad_outp, const float* soft_attn_w_tr, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_val);
extern "C" __global__ void grad_soft_attn_w(const float* grad_outp, const float* values_tr, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_soft_attn_ws);
extern "C" __global__ void grad_attn_w(const float* grad_soft_attn_ws, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, float* grad_attn_ws);
extern "C" __global__ void grad_q(const float* grad_attn_ws, const float* keys, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_queries);
extern "C" __global__ void grad_k(const float* grad_attn_ws_tr, const float* queries, const int n_heads, const int batch_size, const int spat1, const int* start_inds, const int* sizes, const int spat2, const int feature_dims, float* grad_keys);

#endif