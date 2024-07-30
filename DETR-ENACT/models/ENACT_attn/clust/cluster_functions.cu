#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "clust_func.h"
using namespace std;


__global__ void sum_groups_kernel(const float* entropy, const float* query, float* clust_query, const int* group_sizes, const int* group_start_indices, int numCols, int numGroups) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < numGroups && col < numCols) {
        int group_start = group_start_indices[row];
        int group_size = group_sizes[row];

        float sum = 0.0;
        float sum_exp = 0.0;
        
        for (int i = 0; i < group_size; i++) {
        	int ent_idx = group_start + i;
            int idx = ent_idx * numCols + col;
            sum += exp(entropy[ent_idx])*query[idx];
            sum_exp += exp(entropy[ent_idx]);
        }

        int c_index = row * numCols + col;
        clust_query[c_index] = sum/sum_exp;
    }
}

/*
__global__ void sum_groups_kernel(const float* entropy, const float* query, float* clust_query, const int* group_sizes, const int* group_start_indices, int numDims, int numPixels){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < numPixels && col < numDims){
        float sum = 0.0;
        float sum_exp = 0.0;

        for (int i=0; i<)
    }
}
*/
at::Tensor SumGroups(at::Tensor entropy, at::Tensor entropy_step, at::Tensor query) {
    int numRows = query.size(0);
    int numCols = query.size(1);

    // Convert input tensors to CPU for group identification
    auto entropy_step_cpu = entropy_step.to(torch::kCPU);
    const float* entropy_step_ptr = entropy_step_cpu.data_ptr<float>();

    vector<int> group_sizes;
    vector<int> group_start_indices;
    int current_group_size = 1;
    int current_group_start = 0;

    for (int i = 1; i < numRows; i++) {
        if (entropy_step_ptr[i] == entropy_step_ptr[i - 1]) {
            current_group_size++;
        } else {
            group_sizes.push_back(current_group_size);
            group_start_indices.push_back(current_group_start);
            current_group_size = 1;
            current_group_start = i;
        }
    }

    group_sizes.push_back(current_group_size);
    group_start_indices.push_back(current_group_start);

    int numGroups = group_sizes.size();

    // Allocate and copy group_sizes and group_start_indices to the GPU
    int* group_sizes_gpu;
    int* group_start_indices_gpu;
    cudaMalloc(&group_sizes_gpu, numGroups * sizeof(int));
    cudaMalloc(&group_start_indices_gpu, numGroups * sizeof(int));
    cudaMemcpy(group_sizes_gpu, group_sizes.data(), numGroups * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_start_indices_gpu, group_start_indices.data(), numGroups * sizeof(int), cudaMemcpyHostToDevice);

    // Create output tensor on the GPU
    at::Tensor query_cl = at::zeros({numGroups, numCols}, query.options());

    // Launch kernel
    dim3 threadsPerBlock(numCols);
    dim3 numBlocks(numGroups);
    sum_groups_kernel<<<numBlocks, threadsPerBlock>>>(entropy.data_ptr<float>(), query.data_ptr<float>(), query_cl.data_ptr<float>(), group_sizes_gpu, group_start_indices_gpu, numCols, numGroups);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(group_sizes_gpu);
    cudaFree(group_start_indices_gpu);

    return query_cl;

}