#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

extern "C" __global__ void attention_weights(const float* queries, const float* keys, const int n_heads, const int batch_size, const int spatial_sizes_uncl,
                                  const int* spatial_start_ind_cl, const int* spatial_sizes_cl, const int sum_cl_pixels, const int feat_dims, float* attn_w);

extern "C" __global__ void softmax(const float* attn_ws, const int batch_size, const int n_heads, const int spatial_1, const int spatial_2, const int* spatial_start_ind_cl, const int* spatial_sizes_cl, float* soft_attn_w);
extern "C" __global__ void attention(const float* attn_w, const float* values, const int n_heads, const int batch_size, const int spatial_1, const int feat_dims, const int spatial_2, float* attn);
extern "C" __global__ void dot_product(const float* tensor1, const float* tensor2, const int n_heads, const int batch_size, const int spatial_dim1, const int spatial_dim2, const int feat_dims, float* result);
extern "C" __global__ void create_Jacobian(const float* tensor, const int Dim1, const int Dim2, const int Dim3, float* J);

vector<at::Tensor> forward_mhsa(at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes);
vector<at::Tensor> backward_mhsa(at::Tensor grad_attn, at::Tensor attn_w, at::Tensor Queries, at::Tensor Keys, at::Tensor Values, vector<int> clust_start_inds, vector<int> clust_sizes, int total_cl_size);