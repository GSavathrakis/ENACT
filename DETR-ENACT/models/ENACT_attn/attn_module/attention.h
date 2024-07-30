#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

/*vector<at::Tensor> forward_mhsa(at::Tensor query, at::Tensor key, at::Tensor value, int n_heads, vector<int> cluster_end_inds_cumsum, vector<int> n_clusters, vector<int> n_clusters_cumsum, 
                        vector<int> cluster_end_inds_times_spat, vector<int> cluster_start_inds_times_spat_cumsum, vector<int> cluster_end_inds_times_spat_all_pixs_attn, vector<int> cluster_start_inds_times_spat_all_pixs_attn,
                        vector<int> cluster_end_inds_times_spat_all_pixs_vals, vector<int> cluster_start_inds_times_spat_all_pixs_vals, vector<int> group_sizes);

vector<at::Tensor> backward_mhsa(at::Tensor grad_out, at::Tensor attn_w, at::Tensor Value, at::Tensor Key, at::Tensor Query, vector<int> cluster_start_inds_all_pixs_all_inds_vals, vector<int> cluster_end_inds_times_spat_all_pixs_attn, vector<int> group_sizes, int total_spat_cl, 
                                 vector<int> cluster_end_inds_cumsum, vector<int> n_clusters, vector<int> n_clusters_cumsum, vector<int> cluster_start_inds_times_spat_all_pixs_attn, vector<int> group_sizes_all_pixs,
                                 vector<int> cluster_end_inds_times_spat_all_pixs_vals, vector<int> cluster_start_inds_times_spat_all_pixs_vals);*/
//vector<at::Tensor> backward_mhsa(at::Tensor tens1, at::Tensor tens2, at::Tensor tens3, int n_heads, list<int> n_clusters, at::Tensor grad_output);

at::Tensor forward_mhsa(at::Tensor query, at::Tensor key, at::Tensor value, int n_heads, list<int> n_clusters);
vector<at::Tensor> backward_mhsa(at::Tensor tens1, at::Tensor tens2, at::Tensor tens3, int n_heads, list<int> n_clusters, at::Tensor grad_output);