#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
//#include <ATen/ATen.h>
#include <torch/extension.h>
#include "attention.h"
#include "ops/ops.h"

void grad_attention(const torch::Tensor grad_output, const torch::Tensor Values, const torch::Tensor clust_start_inds, const torch::Tensor clust_sizes, 
                    torch::Tensor grad_soft_attn_ws){

}