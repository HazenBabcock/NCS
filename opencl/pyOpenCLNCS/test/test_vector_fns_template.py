#!/usr/bin/env python
#
# Test vector functions generated from templates.
#
# Hazen 08/19
#
import numpy
import os
import pyopencl as cl

import pyOpenCLNCS.template.template_arguments as targs
import pyOpenCLNCS.template.vector_functions as vf

#
# OpenCL setup.
#

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Create context and command queue
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context,
                        properties=cl.command_queue_properties.PROFILING_ENABLE)

# Open program file and build
def buildProgram(code):
    program = cl.Program(context, code)
    try:
        program.build()
    except:
        print("Build log:")
        print(program.get_build_info(devices[0], 
                                     cl.program_build_info.LOG))
        raise

    return program


n_reps = 1


#
# veccopy()
#

veccopy_kernel_code = """

__kernel void veccopy_test(__global float *g_v1,
                           __global float *g_v2)
{
  int i;
  int lid = get_local_id(0);

  __local float v1[256];
  __local float v2[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v2[i] = g_v2[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  veccopy(v1, v2, lid);

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0){
    for(i=0;i<256;i++){
      g_v1[i] = v1[i];
    }
  }
}

"""

def test_veccopy():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.veccopy("v1", "v2", args)
        program = buildProgram(vf_code + veccopy_kernel_code)

        for i in range(n_reps):
            v1 = numpy.zeros(256, dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)

            v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            
            program.veccopy_test(queue, (size,), (size,), v1_buffer, v2_buffer)
            cl.enqueue_copy(queue, v1, v1_buffer).wait()
            queue.finish()

            assert numpy.allclose(v1, v2)


#
# vecncopy()
#

vecncopy_kernel_code = """

__kernel void vecncopy_test(__global float *g_v1,
                            __global float *g_v2)
{
  int i;
  int lid = get_local_id(0);

  __local float v1[256];
  __local float v2[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v2[i] = g_v2[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  vecncopy(v1, v2, lid);

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0){
    for(i=0;i<256;i++){
      g_v1[i] = v1[i];
    }
  }
}

"""

def test_vecncopy():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.vecncopy("v1", "v2", args)
        program = buildProgram(vf_code + vecncopy_kernel_code)

        for i in range(n_reps):
            v1 = numpy.zeros(256, dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)

            v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            
            program.vecncopy_test(queue, (size,), (size,), v1_buffer, v2_buffer)
            cl.enqueue_copy(queue, v1, v1_buffer).wait()
            queue.finish()

            assert numpy.allclose(v1, -v2)


#
# vecdot()
#

vecdot_kernel_code = """

__kernel void vecdot_test(__global float *g_v1,
                          __global float *g_v2,
                          __global float *g_v3)
{
  int i;
  int lid = get_local_id(0);

  __local float v1[256];
  __local float v2[256];
  __local float v3[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v2[i] = g_v1[i];
      v3[i] = g_v2[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  vecdot(v1, v2, v3, lid);

  if (lid == 0){
    *g_v3 = v1[0];
  }
}

"""

def test_vecdot():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.vecdot("v1", "v2", "v3", args)
        program = buildProgram(vf_code + vecdot_kernel_code)

        for i in range(n_reps):
            v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v3 = numpy.zeros(1).astype(numpy.float32)

            v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
            
            program.vecdot_test(queue, (size,), (size,), v1_buffer, v2_buffer, v3_buffer)
            cl.enqueue_copy(queue, v3, v3_buffer).wait()
            queue.finish()

            assert numpy.allclose(v3, numpy.sum(v1*v2))
            
            
#
# vecisEqual()
#

vecisEqual_kernel_code = """

__kernel void vecisEqual_test(__global float *g_v1,
                              __global float *g_v2,
                              __global int *g_v3)
{
  int i;
  int lid = get_local_id(0);

  __local int v1[256];
  __local float v2[256];
  __local float v3[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v2[i] = g_v1[i];
      v3[i] = g_v2[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  vecisEqual(v1, v2, v3, lid);

  if (lid == 0){
    *g_v3 = v1[0];
  }
}

"""

def test_vecisEqual():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.vecisEqual("v1", "v2", "v3", args)
        program = buildProgram(vf_code + vecisEqual_kernel_code)

        for i in range(n_reps):
            v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v3 = numpy.zeros(1).astype(numpy.int32)

            v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
            
            program.vecisEqual_test(queue, (size,), (size,), v1_buffer, v2_buffer, v3_buffer)
            cl.enqueue_copy(queue, v3, v3_buffer).wait()
            queue.finish()

            assert (v3[0] == 0)

            v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v2 = numpy.copy(v1)
            v3 = numpy.zeros(1).astype(numpy.int32)

            v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
            
            program.vecisEqual_test(queue, (size,), (size,), v1_buffer, v2_buffer, v3_buffer)
            cl.enqueue_copy(queue, v3, v3_buffer).wait()
            queue.finish()

            assert (v3[0] == 1)


#
# vecfma()
#

vecfma_kernel_code = """

__kernel void vecfma_test(__global float *g_v1,
                          __global float *g_v2,
                          __global float *g_v3,
                          float g_s1)
{
  int i;
  int lid = get_local_id(0);

  __local float v1[256];
  __local float v2[256];
  __local float v3[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v2[i] = g_v2[i];
      v3[i] = g_v3[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  vecfma(v1, v2, v3, g_s1, lid);

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0){
    for(i=0;i<256;i++){
      g_v1[i] = v1[i];
    }
  }
}

"""

def test_vecfma():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.vecfma("v1", "v2", "v3", "v4", args)
        program = buildProgram(vf_code + vecfma_kernel_code)

        for i in range(n_reps):
            v1 = numpy.zeros(256).astype(dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v3 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v4 = numpy.float32(numpy.random.uniform())

            v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            v3_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
            
            program.vecfma_test(queue, (size,), (size,), v1_buffer, v2_buffer, v3_buffer, v4)
            cl.enqueue_copy(queue, v1, v1_buffer).wait()
            queue.finish()
                
            assert (numpy.allclose(v1, v2*v4 + v3))


#
# vecfmaInplace()
#

vecfmaInplace_kernel_code = """

__kernel void vecfmaInplace_test(__global float *g_v1,
                                 __global float *g_v2,
                                 float g_s1)
{
  int i;
  int lid = get_local_id(0);

  __local float v1[256];
  __local float v2[256];

  if (lid == 0){
    for(i=0;i<256;i++){
      v1[i] = g_v1[i];
      v2[i] = g_v2[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  vecfmaInplace(v1, v2, g_s1, lid);

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0){
    for(i=0;i<256;i++){
      g_v1[i] = v1[i];
    }
  }
}

"""

def test_vecfmaInplace():
    
    for size in targs.size_to_depth:
        args = targs.arguments(work_group_size = size)
        targs.addOpenCL(args)
        
        vf_code = vf.vecfmaInplace("v1", "v2", "v3", args)
        program = buildProgram(vf_code + vecfmaInplace_kernel_code)

        for i in range(n_reps):
            v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = 256).astype(dtype = numpy.float32)
            v3 = numpy.float32(numpy.random.uniform())

            v1_c = numpy.copy(v1)
            
            v1_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
            v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
            
            program.vecfmaInplace_test(queue, (size,), (size,), v1_buffer, v2_buffer, v3)
            cl.enqueue_copy(queue, v1, v1_buffer).wait()
            queue.finish()
                
            assert (numpy.allclose(v1, v2*v3 + v1_c))

            
if (__name__ == "__main__"):
    test_vecfmaInplace()

    
