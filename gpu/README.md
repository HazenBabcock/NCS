## NCS on a GPU using OpenCL or CUDA ##

An OpenCL and CUDA GPU kernels for NCS. The core code is in the `ncs_core.c`
file. Depending on whether you are targeting OpenCL or CUDA you will need to
prepend the code in this file to either the `ncs_opencl.cl` file or the
`ncs_cuda.cu` file.

The tests in the `NCS/gpu/pyOpenCLNCS/test` folder are much more comprehensive
than those in the `NCS/gpu/pyCUDANCS/test`, so these are the ones to run if
something seems off.

## Performance ##

Tests were done using the `NCS/gpu/pyOpenCLNCS/profile.py` Python script.
Speedup is relative to the clib version of NCS on the same computer.

* Nvidia Tesla K20Xm - 5x speedup.
* Intel Haswell-ULT Integrated Graphics Controller - 4.2x speedup.
* Nvidia GeForce GT 1030 - 1.2x speedup.

## Example Usage ##

Please see the Jupyter notebooks in the `jupyter_notebooks` folder for
examples of how to use the kernels.

### Jupyter notebook dependencies ###

#### Python 3 ####

* [numpy](http://www.numpy.org/)
* [pyopencl](https://documen.tician.de/pyopencl/) (for OpenCL)
* [pycuda](https://documen.tician.de/pycuda/) (for CUDA)
