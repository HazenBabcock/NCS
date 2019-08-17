#!/usr/bin/env python
#
# Used for quickly measuring how long the solver takes to run.
#
# Hazen 08/19
#
import numpy
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule
import time

# python3 and C NCS reference version.
import pyCNCS.ncs_c as ncsC

# Need this for the OTF mask.
import pyOpenCLNCS
import pyOpenCLNCS.py_ref as pyRef


import pyCUDANCS

#
# CUDA setup.
#
kernel_code = pyCUDANCS.loadNCSKernel()

mod = SourceModule(kernel_code)
ncs_fn = mod.get_function("ncsReduceNoise")


alpha = 0.1
n_pts = 16


def profile(n_reps):
   """
   Report how long it takes to reduce the noise in X sub-regions.
   """
   
   # Setup
   numpy.random.seed(1)
   
   data = numpy.random.uniform(low = 10.0, high = 20.0, size = (n_reps, n_pts, n_pts)).astype(dtype = numpy.float32)
   gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
   otf_mask_shift = pyRef.createOTFMask()

   # CUDA Setup.
   u = numpy.zeros((n_reps, n_pts, n_pts), dtype = numpy.float32)
   iters = numpy.zeros(n_reps, dtype = numpy.int32)
   status = numpy.zeros(n_reps, dtype = numpy.int32)

   # Run CUDA noise reduction kernel on the sub-regions.
   start_time = time.time()
   ncs_fn(drv.In(data_in),
          drv.In(gamma),
          drv.In(self.otf_mask),
          drv.Out(data_out),
          drv.Out(iters),
          drv.Out(status),
          block = (num_sr, 1, 1),
          grid = (1,1))
   e_time = time.time() - start_time

   e_time = 1.0e-9*(ev1.profile.end - ev1.profile.start)
   print("CUDA {0:.6f} seconds".format(e_time))


def profileNCSC(n_reps):
   """
   The C reference version for comparison.
   """
   numpy.random.seed(1)

   data = numpy.random.uniform(low = 10.0, high = 20.0, size = (n_reps, n_pts, n_pts)).astype(dtype = numpy.float32)
   gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
   otf_mask = numpy.fft.fftshift(pyRef.createOTFMask().reshape(16, 16))

   ref_u = numpy.zeros_like(data)

   ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)

   start_time = time.time()
   for i in range(n_reps):
      ncs_sr.newRegion(data[i,:,:], gamma)
      ncs_sr.setOTFMask(otf_mask)
      ref_u[i,:,:] = ncs_sr.cSolve(alpha, verbose = False)
   e_time = time.time() - start_time

   ncs_sr.cleanup()
   print("CNSC {0:.6f} seconds".format(e_time))
    

if (__name__ == "__main__"):
   import argparse

   parser = argparse.ArgumentParser(description = 'NCS in CUDA')

   parser.add_argument('--reps', dest='reps', type=int, required=False, default = 1000,
                       help = "Number sub-regions to process in profiling.")
   args = parser.parse_args()
   
   profile(args.reps)
   profileNCSC(args.reps)
