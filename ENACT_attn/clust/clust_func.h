#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <torch/torch.h>
using namespace std;

// Declaration of the function that will be defined in kernels.cu
at::Tensor SumGroups(at::Tensor entropy, at::Tensor entropy_step, at::Tensor query);

// Declaration of the function that will be defined in functions.cpp
list<at::Tensor> enact_cluster(at::Tensor entropy, at::Tensor entropy_step, at::Tensor query);
list<int> n_clusters(at::Tensor entropy_step);

#endif // FUNCTIONS_H