#include "clust/clust_func.h"
#include "attn_module/attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enact_cluster_forward", &enact_cluster_forward, "A function that sums rows of a 2D tensor based on consecutive groups in a 1D tensor using CUDA and returns the result");
    m.def("enact_cluster_backward", &enact_cluster_backward, "A function that sums rows of a 2D tensor based on consecutive groups in a 1D tensor using CUDA and returns the result");
    m.def("forward_mhsa", &forward_mhsa, "Forward pass of the clustered attention module");
    m.def("backward_mhsa", &backward_mhsa, "Backward pass of the clustered attention module");
}