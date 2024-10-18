#include "clust_func.h"
#include <iostream>
#include <list>
using namespace std;

list<at::Tensor> enact_cluster(at::Tensor entropy, at::Tensor entropy_step, at::Tensor src_tensor) {
    // Ensure tensors are on the CUDA device
    auto entropy_cuda = entropy.to(torch::kCUDA);
    auto entropy_step_cuda = entropy_step.to(torch::kCUDA);
    auto src_tensor_cuda = src_tensor.to(torch::kCUDA);

    int bs = src_tensor.size(0);

    list<at::Tensor> tensors_list;

    for (int n=0; n<bs; n++){
    	at::Tensor clustered_src_tensor_cuda = SumGroups(entropy_cuda[n], entropy_step_cuda[n], src_tensor_cuda[n]);
    	tensors_list.push_back(clustered_src_tensor_cuda);
    }

    return tensors_list;
}

list<int> n_clusters(at::Tensor entropy_step){
    auto entropy_step_cpu = entropy_step.to(torch::kCPU);

    int bs = entropy_step_cpu.size(0);
    int spat = entropy_step_cpu.size(1);

    list<int> num_clusters;

    for (int n=0; n<bs; n++){
        int n_cl = 1;
        for (int d=1; d<spat; d++){
            if (entropy_step_cpu.index({n,d}).item<float>()!=entropy_step_cpu.index({n,d-1}).item<float>()){
                n_cl++;
            }
        }
        num_clusters.push_back(n_cl);
    }
    return num_clusters;
}