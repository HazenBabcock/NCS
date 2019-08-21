#!/usr/bin/env python
#
# CUDA Python wrapper.
#
# Hazen 08/19
#

import os

def loadNCSKernel():
    kernel_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ncs.cu")
    with open(kernel_filename) as fp:
        kernel_code = fp.read()

    return kernel_code
