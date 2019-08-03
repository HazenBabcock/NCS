#!/usr/bin/env python
#
# Pure Python reference version for testing.
#
# Written to match the OpenCL code, not for style..
#
# Unfortunately  the match is not perfect. I tried to use numpy.float32
# for all the math but there are still small differences..
#
# Hazen 08/19
#

import numpy

## Constants
FITMIN = numpy.float32(1.0e-6)
PSIZE = 64

C_1  = numpy.float32(1.0e-4)
EPSILON = numpy.float32(1.0e-5)
MAXITERS = 200

MIN_STEP = numpy.float32(1.0e-6)
STEPM = numpy.float32(0.5)

## Error status codes.
SUCCESS = 0
UNSTARTED = -1
REACHED_MAXITERS = -2
INCREASING_GRADIENT = -3
MINIMUM_STEP = -4


## FFT functions.

def fft_16x16(x_r, x_c, y_r, y_c):
    t1 = x_r.reshape(16,16)
    t2 = x_c.reshape(16,16)
    x_fft = numpy.fft.fft2(t1 + 1j*t2)
    y_r[:] = numpy.real(x_fft).flatten()
    y_c[:] = numpy.imag(x_fft).flatten()

    
## Vector functions.

def veccopy(v1, v2):
    v1[:] = v2

def vecncopy(v1, v2):
    v1[:] = -v2

def vecdot(v1, v2):
    return numpy.sum(v1 * v2)

def vecfma(v1, v2, v3, s1):
    v1[:] = v2*s1 + v3

def vecfmaInplace(v1, v2, s1):
    v1[:] = v2*s1 + v1

def vecmul(v1, v2, v3):
    v1[:] = v2*v3

def vecnorm(v1):
    return numpy.sqrt(vecdot(v1, v1))

def vecscaleInplace(v1, s1):
    v1[:] = v1*s1

def vecsub(v1, v2, v3):
    v1[:] = v2 - v3


## NCS functions.

def calcLLGradient(u, data, gamma, gradient):
    t1 = data + gamma
    t2 = u + gamma
    t2[(t2 < FITMIN)] = FITMIN
    gradient[:] = 1.0 - t1/t2

def calcLogLikelihood(u, data, gamma):
    t1 = data + gamma
    t2 = u + gamma
    t2[(t2 < FITMIN)] = FITMIN
    t2 = numpy.log(t2)
    
    return numpy.sum(u - t1*t2)

def calcNCGradient(u_fft_grad_r, u_fft_grad_c, u_fft_r, u_fft_c, otf_mask_sqr, gradient):
    
    for i in range(PSIZE*4):
        t1 = u_fft_r * u_fft_grad_r[i] + u_fft_c * u_fft_grad_c
        t1 = 2.0 * t1 * otf_mask_sqr
        gradient[i] = numpy.sum(t1)/(4.0 * PSIZE)

def calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr):
    t1 = u_fft_r * u_fft_r + u_fft_c * u_fft_c
    t1 = t1 * otf_mask_sqr
    return numpy.sum(t1)/(4.0 * PSIZE)

def converged(x, g):
    xnorm = max(vecnorm(x), 1.0)
    gnorm = vecnorm(g)
    if ((gnorm/xnorm) > EPSILON):
        return False
    else:
        return True

def initUFFTGrad(u_fft_grad_r, u_fft_grad_c):
    x_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    x_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    y_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    y_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
                
    for i in range(4*PSIZE):
        x_r[i] = 1.0
        fft16x16(x_r, x_c, y_r, y_c)
        u_fft_grad_r[i] = y_r
        u_fft_grad_c[i] = y_c
        x_r[i] = 0.0
