from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ENACT',
    ext_modules=[
        CUDAExtension(
            name='ENACT',
            sources=['clust/clust_func.cpp', 'clust/cluster_functions.cu', 'attn_module/attention.cu', 'bindings_py.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)