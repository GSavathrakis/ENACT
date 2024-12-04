from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ENACT',
    ext_modules=[
        CUDAExtension(
            name='ENACT',
            sources=['clust/cluster_functions.cu', 'attn_module/attv_prod.cu', 'attn_module/grad_attn.cu', 'attn_module/grad_k.cu', 'attn_module/grad_q.cu', 'attn_module/grad_v.cu', 'attn_module/jacobian.cu', 'attn_module/qk_prod.cu', 'attn_module/softmax.cu', 'attn_module/ops/ops.cu', 'bindings_py.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)