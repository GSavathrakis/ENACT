#include "clust/clust_func.h"
#include "attn_module/attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enact_cluster_forward", &enact_cluster_forward, "A function that sums rows of a 2D tensor based on consecutive groups in a 1D tensor using CUDA and returns the result");
    m.def("enact_cluster_backward", &enact_cluster_backward, "A function that computes the grad of the previous function");
    m.def("enact_cluster_forward_rcda", &enact_cluster_forward_rcda, "A function that sums rows of a 2D tensor based on consecutive groups in a 1D tensor using CUDA and returns the result");
    m.def("enact_cluster_backward_rcda", &enact_cluster_backward_rcda, "A function that computes the grad of the previous function");
    m.def("forward_mhsa", &forward_mhsa, "Forward pass of the clustered attention module");
    m.def("backward_mhsa", &backward_mhsa, "Backward pass of the clustered attention module");
    m.def("forward_rcda_w", &forward_rcda_w, "Forward pass to compute RCDA weights");
    m.def("backward_rcda_w", &backward_rcda_w, "Backward pass from RCDA weights gradient");
    m.def("forward_rcda_map", &forward_rcda_map, "Forward pass to compute RCDA attention map");
    m.def("backward_rcda_map", &backward_rcda_map, "Backward pass from RCDA attention map gradient");
}