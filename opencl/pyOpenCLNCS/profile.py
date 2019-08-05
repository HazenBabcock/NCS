#!/usr/bin/env python
#
# Used for quickly measuring how long the solver takes to run.
#
# Hazen 08/19
#
import numpy
import pyopencl as cl

import pyOpenCLNCS


#
# OpenCL setup.
#
kernel_code = pyOpenCLNCS.loadNCSKernel()

# Create context and command queue
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context,
                        properties=cl.command_queue_properties.PROFILING_ENABLE)

# Open program file and build
program = cl.Program(context, kernel_code)
try:
   program.build()
except:
   print("Build log:")
   print(program.get_build_info(devices[0], 
         cl.program_build_info.LOG))
   raise

alpha = 0.1
n_pts = 16

def profile():

   # Setup
   numpy.random.seed(1)
   n_reps = 1000

   data = numpy.random.uniform(low = 10.0, high = 20.0, size = (n_reps, n_pts, n_pts)).astype(dtype = numpy.float32)
   gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
   otf_mask = numpy.random.uniform(size = (n_pts, n_pts)).astype(numpy.float32)
   otf_mask_shift = numpy.fft.fftshift(otf_mask)

   # OpenCL Setup.   
   u_fft_grad_r = numpy.zeros((n_pts * n_pts, n_pts, n_pts)).astype(numpy.float32)
   u_fft_grad_c = numpy.zeros((n_pts * n_pts, n_pts, n_pts)).astype(numpy.float32)
   
   u_fft_grad_r_buffer = cl.Buffer(context, 
                                   cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                   hostbuf = u_fft_grad_r)
   u_fft_grad_c_buffer = cl.Buffer(context, 
                                   cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                   hostbuf = u_fft_grad_c)
   
   u = numpy.zeros((n_reps, n_pts, n_pts), dtype = numpy.float32)
   iters = numpy.zeros(n_reps, dtype = numpy.int32)
   status = numpy.zeros(n_reps, dtype = numpy.int32)
   
   data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                           hostbuf = data)
   gamma_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                            hostbuf = gamma)
   otf_mask_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                               hostbuf = otf_mask_shift)
   u_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                        hostbuf = u)
   iters_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                            hostbuf = iters)
   status_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                             hostbuf = status)

   # OpenCL noise reduction.
   ev1 = program.initUFFTGrad(queue, (1,), (1,), u_fft_grad_r_buffer, u_fft_grad_c_buffer)

   ev2 = program.ncsReduceNoise(queue, (n_reps,), (1,),
                                u_fft_grad_r_buffer,
                                u_fft_grad_c_buffer,
                                data_buffer,
                                gamma_buffer,
                                otf_mask_buffer,
                                u_buffer,
                                iters_buffer,
                                status_buffer,
                                numpy.float32(alpha))

   cl.enqueue_copy(queue, u, u_buffer).wait()
   cl.enqueue_copy(queue, iters, iters_buffer).wait()
   cl.enqueue_copy(queue, status, status_buffer).wait()
   queue.finish()

   e_time = 1.0e-9*(ev1.profile.end - ev1.profile.start)
   e_time += 1.0e-9*(ev2.profile.end - ev2.profile.start)
   print("{0:.6f} seconds".format(e_time))


if (__name__ == "__main__"):
    profile()
    
