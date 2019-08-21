## NCS on a GPU using CUDA ##

An CUDA GPU kernel for NCS.

## Note ##

This was ported from OpenCL. The OpenCL equivalent has much more
rigorous tests.

## Performance ##

Tests were done using the `NCS/cuda/pyCUDANCS/profile.py` Python script.
Speedup is relative to the clib version of NCS on the same computer.

* Nvidia Tesla K20Xm - X speedup.
* Intel Haswell-ULT Integrated Graphics Controller - X speedup.
* Nvidia GeForce GT 1030 - X speedup.
