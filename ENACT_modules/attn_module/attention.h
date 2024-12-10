#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

vector<torch::Tensor> forward_mhsa(const torch::Tensor Queries, const torch::Tensor Keys, const torch::Tensor Values, const torch::Tensor clust_start_inds, 
                  const torch::Tensor clust_sizes);
vector<torch::Tensor> backward_mhsa(const torch::Tensor grad_output, const torch::Tensor soft_attn_ws, const torch::Tensor Queries, const torch::Tensor Keys, 
                   const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes);

/*void qk_product(const torch::Tensor Queries, const torch::Tensor Keys, const torch::Tensor clust_start_inds, 
                const torch::Tensor clust_sizes, const int num_heads, const int batch_size, const int spatial_size, 
                const int feature_dims, const int concat_spatial_dims, torch::Tensor attn_ws);
void soft(const torch::Tensor attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, const int num_heads, const int batch_size, 
          const int spatial_dims, const int concat_spatial_dims, torch::Tensor soft_attn_ws);
void attn(const torch::Tensor soft_attn_ws, const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
          const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor attn);

void grad_Values(const torch::Tensor grad_output, const torch::Tensor soft_attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
                 const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_values);
void grad_Attention(const torch::Tensor grad_output, const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
                    const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_soft_attn_ws);
void Jacobian(const torch::Tensor grad_soft_attn_ws, const torch::Tensor soft_attn_ws, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
              const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, torch::Tensor grad_attn_ws);
void grad_Queries(const torch::Tensor grad_attn_ws, const torch::Tensor Keys, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
                  const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_queries);
void grad_Keys(const torch::Tensor grad_attn_ws, const torch::Tensor Queries, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes,
               const int num_heads, const int batch_size, const int spatial_dims, const int concat_spatial_dims, const int feature_dims, torch::Tensor grad_keys);*/
#endif // FUNCTIONS_H