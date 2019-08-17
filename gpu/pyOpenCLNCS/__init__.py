#!/usr/bin/env python
#
# OpenCL Python wrapper.
#
# Hazen 08/19
#

import os

def loadNCSKernel():
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    kernel_code = ""
    kernel_filename = os.path.join(code_dir, "ncs_core.c")
    with open(kernel_filename) as fp:
        kernel_code += fp.read()

    kernel_filename = os.path.join(code_dir, "ncs_opencl.cl")
    with open(kernel_filename) as fp:
        kernel_code += fp.read()

    return kernel_code
