/*
 * Core GPU code for NCS. This is shared between the OpenCL and CUDA implementations. I
 * wasn't sure what the best extension was for this file.
 *
 * Key resources that I used in this implementation:
 *
 * 1) The book "OpenCL In Action" by Matthew Scarpino.
 * 2) The C port of L-BFGS by Naoaki Okazaki (http://www.chokkan.org/software/liblbfgs/).
 * 3) Inexact line search conditions (https://en.wikipedia.org/wiki/Wolfe_conditions).
 *
 * Hazen 08/19
 */

/*
 * License for the Limited memory BFGS (L-BFGS) solver code.
 *
 * Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* Threshold for handling negative values in the fit. */
#define FITMIN 1.0e-6f

/*
 * The problem size is (16*16)/4 or 256/4.
 *
 * This is changeable to make easier at some point in the future
 * to use a different ROI size. However this would also involve writing
 * the appropriate 2D FFT.
 */ 
#define PSIZE 64

/* L-BFGS solver parameters. */

#define C_1 1.0e-4f        /* Armijo rule/condition scaling value. */
#define EPSILON 1.0e-4f    /* Stopping point. */
#define M 8                /* Number of history points saved. Must be a power of 2. */
#define MAXITERS 200       /* Maximum number of iterations. */
#define MIN_STEP 1.0e-6f   /* Minimum step size. */
#define STEPM 0.5          /* Step size multiplier. */

/* Error status codes. */
#define SUCCESS 0
#define UNSTARTED -1
#define REACHED_MAXITERS -2
#define INCREASING_GRADIENT -3
#define MINIMUM_STEP -4
#define REACHED_MAXPRECISION -5


/****************
 * FFT functions.
 ****************/

// 4 point complex FFT
void fft4(float4 x_r, float4 x_c, float4 *y_r, float4 *y_c)
{
    float t1_r = x_r.x + x_r.z;
    float t1_c = x_c.x + x_c.z;
    
    float t2_r = x_r.x - x_r.z;
    float t2_c = x_c.x - x_c.z;

    float t3_r = x_r.y + x_r.w;
    float t3_c = x_c.y + x_c.w;

    float t4_r = x_r.y - x_r.w;
    float t4_c = x_c.y - x_c.w;
 
    y_r[0].s0 = t1_r + t3_r;
    y_r[0].s1 = t2_r + t4_c;
    y_r[0].s2 = t1_r - t3_r;
    y_r[0].s3 = t2_r - t4_c;

    y_c[0].s0 = t1_c + t3_c;
    y_c[0].s1 = t2_c - t4_r;
    y_c[0].s2 = t1_c - t3_c;
    y_c[0].s3 = t2_c + t4_r;
}


// 4 point complex IFFT
void ifft4(float4 x_r, float4 x_c, float4 *y_r, float4 *y_c)
{
    float t1_r = x_r.x + x_r.z;
    float t1_c = x_c.x + x_c.z;
    
    float t2_r = x_r.x - x_r.z;
    float t2_c = x_c.x - x_c.z;

    float t3_r = x_r.y + x_r.w;
    float t3_c = x_c.y + x_c.w;

    float t4_r = x_r.y - x_r.w;
    float t4_c = x_c.y - x_c.w;
 
    y_r[0].s0 = t1_r + t3_r;
    y_r[0].s1 = t2_r - t4_c;
    y_r[0].s2 = t1_r - t3_r;
    y_r[0].s3 = t2_r + t4_c;

    y_c[0].s0 = t1_c + t3_c;
    y_c[0].s1 = t2_c + t4_r;
    y_c[0].s2 = t1_c - t3_c;
    y_c[0].s3 = t2_c - t4_r;
    
    y_r[0] = y_r[0]*0.25f;
    y_c[0] = y_c[0]*0.25f;
}


// 8 point complex FFT.
void fft8(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r;
    float4 t_c;
    float4 f41_r;
    float4 f41_c;
    float4 f42_r;
    float4 f42_c;
     
    // 4 point FFT.
    t_r = (float4)(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c = (float4)(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    fft4(t_r, t_c, &f41_r, &f41_c);

    t_r = (float4)(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c = (float4)(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    fft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = (float4)(1.0f, 7.07106781e-01f, 0.0f, -7.07106781e-01f);
    float4 c1 = (float4)(0.0f, 7.07106781e-01f, 1.0f,  7.07106781e-01f);
    y_r[0] = f41_r + f42_r * r1 + f42_c * c1;
    y_c[0] = f41_c + f42_c * r1 - f42_r * c1;
    y_r[1] = f41_r - f42_r * r1 - f42_c * c1;
    y_c[1] = f41_c - f42_c * r1 + f42_r * c1;    
}


// 8 point complex IFFT.
void ifft8(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r;
    float4 t_c;
    float4 f41_r;
    float4 f41_c;
    float4 f42_r;
    float4 f42_c;
     
    // 4 point IFFT.
    t_r = (float4)(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c = (float4)(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    ifft4(t_r, t_c, &f41_r, &f41_c);

    t_r = (float4)(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c = (float4)(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    ifft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = (float4)(1.0f,  7.07106781e-01f,  0.0f, -7.07106781e-01f);
    float4 c1 = (float4)(0.0f, -7.07106781e-01f, -1.0f, -7.07106781e-01f);
    y_r[0] = f41_r + f42_r * r1 + f42_c * c1;
    y_c[0] = f41_c + f42_c * r1 - f42_r * c1;
    y_r[1] = f41_r - f42_r * r1 - f42_c * c1;
    y_c[1] = f41_c - f42_c * r1 + f42_r * c1;
    
    y_r[0] = y_r[0]*0.5f;
    y_c[0] = y_c[0]*0.5f;
    y_r[1] = y_r[1]*0.5f;
    y_c[1] = y_c[1]*0.5f;

}


// 16 point complex FFT.
void fft16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point FFT.
    t_r[0] = (float4)(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = (float4)(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = (float4)(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = (float4)(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    fft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = (float4)(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = (float4)(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = (float4)(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = (float4)(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    fft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = (float4)(1.0f, 9.23879533e-01f, 7.07106781e-01f, 3.82683432e-01f);
    float4 c1 = (float4)(0.0f, 3.82683432e-01f, 7.07106781e-01f, 9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    

    r1 = (float4)(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = (float4)(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;    
}


// 16 point complex IFFT.
void ifft16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point IFFT.
    t_r[0] = (float4)(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = (float4)(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = (float4)(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = (float4)(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    ifft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = (float4)(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = (float4)(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = (float4)(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = (float4)(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    ifft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = (float4)(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    float4 c1 = (float4)(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = (float4)( 0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = (float4)(-1.0f, -9.23879533e-01f, -7.07106781e-01f, -3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;
    
    for(int i = 0; i<4; i++){
        y_r[i] = y_r[i]*0.5f;
        y_c[i] = y_c[i]*0.5f;    
    }
}


// 16 x 16 point complex FFT.
void fft_16x16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t1_r[4];
    float4 t1_c[4];
    
    float *y1_r = (float *)y_r;
    float *y1_c = (float *)y_c;
    
    // Axis 1.
    for(int i=0; i<16; i++){
        fft16(&(x_r[i*4]), &(x_c[i*4]), &(y_r[i*4]), &(y_c[i*4]));
    }
    
    // Axis 2.
    for(int i=0; i<16; i++){
    
        // Convert columns to rows.
        for(int j=0; j<4; j++){
            t1_r[j].x = y1_r[(4*j+0)*16+i];
            t1_r[j].y = y1_r[(4*j+1)*16+i];
            t1_r[j].z = y1_r[(4*j+2)*16+i];
            t1_r[j].w = y1_r[(4*j+3)*16+i];
            t1_c[j].x = y1_c[(4*j+0)*16+i];
            t1_c[j].y = y1_c[(4*j+1)*16+i];
            t1_c[j].z = y1_c[(4*j+2)*16+i];
            t1_c[j].w = y1_c[(4*j+3)*16+i];    
        }
        
        fft16(t1_r, t1_c, t1_r, t1_c);
        
        // Reverse conversion.
        for(int j=0; j<4; j++){
            y1_r[(4*j+0)*16+i] = t1_r[j].x;
            y1_r[(4*j+1)*16+i] = t1_r[j].y;
            y1_r[(4*j+2)*16+i] = t1_r[j].z;
            y1_r[(4*j+3)*16+i] = t1_r[j].w;
            y1_c[(4*j+0)*16+i] = t1_c[j].x;
            y1_c[(4*j+1)*16+i] = t1_c[j].y;
            y1_c[(4*j+2)*16+i] = t1_c[j].z;
            y1_c[(4*j+3)*16+i] = t1_c[j].w;            
        }        
    }
}


// 16 x 16 point complex IFFT.
void ifft_16x16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t1_r[4];
    float4 t1_c[4];

    float *x1_r = (float *)x_r;
    float *x1_c = (float *)x_c;
    float *y1_r = (float *)y_r;
    float *y1_c = (float *)y_c;
    
    // Axis 2.
    for(int i=0; i<16; i++){
    
        // Convert columns to rows.
        for(int j=0; j<4; j++){
            t1_r[j].x = x1_r[(4*j+0)*16+i];
            t1_r[j].y = x1_r[(4*j+1)*16+i];
            t1_r[j].z = x1_r[(4*j+2)*16+i];
            t1_r[j].w = x1_r[(4*j+3)*16+i];
            t1_c[j].x = x1_c[(4*j+0)*16+i];
            t1_c[j].y = x1_c[(4*j+1)*16+i];
            t1_c[j].z = x1_c[(4*j+2)*16+i];
            t1_c[j].w = x1_c[(4*j+3)*16+i];    
        }
        
        ifft16(t1_r, t1_c, t1_r, t1_c);
        
        // Reverse conversion.
        for(int j=0; j<4; j++){
            y1_r[(4*j+0)*16+i] = t1_r[j].x;
            y1_r[(4*j+1)*16+i] = t1_r[j].y;
            y1_r[(4*j+2)*16+i] = t1_r[j].z;
            y1_r[(4*j+3)*16+i] = t1_r[j].w;
            y1_c[(4*j+0)*16+i] = t1_c[j].x;
            y1_c[(4*j+1)*16+i] = t1_c[j].y;
            y1_c[(4*j+2)*16+i] = t1_c[j].z;
            y1_c[(4*j+3)*16+i] = t1_c[j].w;            
        }        
    }
    
    // Axis 1.
    for(int i=0; i<16; i++){
        ifft16(&(y_r[i*4]), &(y_c[i*4]), &(y_r[i*4]), &(y_c[i*4]));
    }     
}


/******************
 * Vector functions.
 ******************/

void veccopy(float4 *v1, float4 *v2)
{
    for(int i=0; i<PSIZE; i++){
        v1[i] = v2[i];
    }
}

void vecncopy(float4 *v1, float4 *v2)
{
    for(int i=0; i<PSIZE; i++){
        v1[i] = -v2[i];
    }
}

float vecdot(float4 *v1, float4 *v2)
{
    float sum = 0;
    
    for(int i=0; i<PSIZE; i++){
        sum += dot(v1[i], v2[i]);
    }
    return sum;
}

int vecisEqual(float4 *v1, float4 *v2)
{
    for(int i=0; i<PSIZE; i++){
    	if (all(isnotequal(v1[i], v2[i]))){
	   return 0;
	}
    }
    return 1; 
}

/* v1 = v2 * s1 + v3 */
void vecfma(float4 *v1, float4 *v2, float4 *v3, float s1)
{
    float4 t1 = (float4)(s1, s1, s1, s1);
    for(int i=0; i<PSIZE; i++){
        v1[i] = fma(t1, v2[i], v3[i]);
    }
}

/* v1 = v1 + v2 * s1 */
void vecfmaInplace(float4 *v1, float4 *v2, float s1)
{
    float4 t1 = (float4)(s1, s1, s1, s1);
    for(int i=0; i<PSIZE; i++){
        v1[i] = fma(t1, v2[i], v1[i]);
    }
}

void vecmul(float4 *v1, float4 *v2, float4 *v3)
{
    for(int i=0; i<PSIZE; i++){
        v1[i] = v2[i]*v3[i];
    }
}

float vecnorm(float4 *v1)
{
    return sqrt(vecdot(v1, v1));
}

void vecscaleInplace(float4 *v1, float s1)
{
    float4 t1 = (float4)(s1, s1, s1, s1);
    for(int i=0; i<PSIZE; i++){
        v1[i] = v1[i]*t1;
    }
}

void vecsub(float4 *v1, float4 *v2, float4 *v3)
{
    for(int i=0; i<PSIZE; i++){
        v1[i] = v2[i] - v3[i];
    }
}


/***********************
 * NCS cost functions.
 ***********************/

void calcLLGradient(float4 *u, float4 *data, float4 *gamma, float4 *gradient)
{
    float4 t1;
    float4 t2;

    for(int i=0; i<PSIZE; i++){
        t1 = data[i] + gamma[i];
        t2 = fmax(u[i] + gamma[i], FITMIN);
        gradient[i] = 1.0f - t1/t2;
    }
}

float calcLogLikelihood(float4 *u, float4 *data, float4 *gamma)
{
    float4 sum = (float4)(0.0, 0.0, 0.0, 0.0);
    float4 t1;
    float4 t2;

    for(int i=0; i<PSIZE; i++){
        t1 = data[i] + gamma[i];
        t2 = log(fmax(u[i] + gamma[i], FITMIN));
        sum += u[i] - t1*t2;
    }
    
    return sum.s0 + sum.s1 + sum.s2 + sum.s3;
}

/* 
 * Use inverse FFT optimization.
 *
 * Notes:
 *
 *  1. Only gives correct values for valid OTFs, meaning
 *     that they are the FT of a realistic PSF.
 *
 *  2. The u_fft_r and u_fft_c parameters must also be the
 *     FT of a real valued image.
 */
void calcNCGradientIFFT(float4 *u_fft_r, 
                        float4 *u_fft_c, 
                        float4 *otf_mask_sqr,
                        float4 *gradient)
{
    int i;
    float4 t1;
    float4 u_r[PSIZE];
    float4 u_c[PSIZE];
    float4 t2[PSIZE];

    for(i=0; i<PSIZE; i++){
        t1 = 2.0f*otf_mask_sqr[i];
        u_r[i] = t1*u_fft_r[i];
        u_c[i] = t1*u_fft_c[i];
    }

    ifft_16x16(u_r, u_c, gradient, t2);
}

float calcNoiseContribution(float4 *u_fft_r, float4 *u_fft_c, float4 *otf_mask_sqr)
{
    float sum = 0.0f;
    float4 t1;
    
    for(int i=0; i<PSIZE; i++){
        t1 = u_fft_r[i]*u_fft_r[i] + u_fft_c[i]*u_fft_c[i];
        t1 = t1*otf_mask_sqr[i];
        sum += t1.s0 + t1.s1 + t1.s2 + t1.s3;
    }

    return sum*(float)(1.0/(4.0*PSIZE));
}


/******************
 * L-BFGS functions.
 ******************/

int converged(float4 *x, float4 *g)
{
    float xnorm = fmax(vecnorm(x), 1.0f);
    float gnorm = vecnorm(g);
    if ((gnorm/xnorm) > EPSILON){
        return 0;
    }
    else{
        return 1;
    }
}

int moduloM(int i)
{
    return i & (M-1);
}

/*****************
 * NCS Function
 *****************/

/*
 * Run NCS noise reduction on a sub-region.
 * 
 * Note: Any zero or negative values in the sub-regions should be
 *       set to a small positive value like 1.0.
 *
 * data - Sub-region data in e-.
 * gamma - Sub-region CMOS variance in units of e-^2.
 * otf_mask_sqr - 16 x 16 array containing the OTF mask.
 * u_r - Noise reduction results.
 * iterations - Number of L-BFGS solver iterations.
 * status - Status of the solution (good, failed because of X).
 * alpha - NCS alpha term.
 */
void ncsReduceNoiseSR(float4 *data,
		      float4 *gamma,
		      float4 *otf_mask_sqr,
		      float4 *u_r,
		      int *iterations,
		      int *status,
		      float alpha)
{
    /* Variables. */
    int i,j,k;
    int bound;
    int ci;
  
    float beta;
    float cost;
    float cost_p;
    float step;
    float ys_c0;
    float yy;
    
    float a[M];
    float ys[M];
    
    float4 g_p[PSIZE];
    float4 gradient[PSIZE];
    float4 srch_dir[PSIZE];
    float4 u_c[PSIZE];
    float4 u_fft_r[PSIZE]; 
    float4 u_fft_c[PSIZE];
    float4 u_p[PSIZE];
    float4 work1[PSIZE];
		      
    float4 s[M][PSIZE];
    float4 y[M][PSIZE];

    /* Initialization. */
    *iterations = 0;
    *status = UNSTARTED;
  
    for (i=0; i<PSIZE; i++){
        u_c[i] = (float4)(0.0, 0.0, 0.0, 0.0);
    }
    
    /* Calculate initial state. */
    fft_16x16(u_r, u_c, u_fft_r, u_fft_c);
    
    /* Cost. */
    cost = calcLogLikelihood(u_r, data, gamma);
    cost += alpha * calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr);
    
    /* Gradient. */
    calcLLGradient(u_r, data, gamma, gradient);
    calcNCGradientIFFT(u_fft_r, u_fft_c, otf_mask_sqr, work1);
    vecfmaInplace(gradient, work1, alpha);

    /* Check if we've already converged. */
    if (converged(u_r, gradient)){
        *iterations = 1;
        *status = SUCCESS;
        return;
    }
    
    /* Initial search direction. */
    step = 1.0/vecnorm(gradient);
    vecncopy(srch_dir, gradient);

    /* Start search. */
    for (k=1; k<(MAXITERS+1); k++){
    
        /* 
         * Line search. 
         *
         * This checks the Armijo rule/condition.
         * https://en.wikipedia.org/wiki/Wolfe_conditions
         */
        float t1 = C_1 * vecdot(srch_dir, gradient);
         
        if (t1 > 0.0){
            /* Increasing gradient. Minimization failed. */
            *iterations = k+1;
            *status = INCREASING_GRADIENT;
            return;
        }
        
        /* Store current cost, u and gradient. */
        cost_p = cost;
        veccopy(u_p, u_r);
        veccopy(g_p, gradient);

	/* Search for a good step size. */        
        int searching = 1;
        while(searching){
        
            /* Move in search direction. */
            vecfma(u_r, srch_dir, u_p, step);
            
            /* Calculate new cost. */
            fft_16x16(u_r, u_c, u_fft_r, u_fft_c);
            cost = calcLogLikelihood(u_r, data, gamma);
            cost += alpha * calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr);
            
            /* Armijo condition. */
            if (cost <= (cost_p + t1*step)){
                searching = 0;
            }
            else{
                step = STEPM*step;
                if (step < MIN_STEP){
                    /* 
                     * Reached minimum step size. Minimization failed. 
                     * Return the last good u values.
                     */
                    for (i=0; i<PSIZE; i++){
                        u_r[i] = u_p[i];

                    }
                    *iterations = k+1;
                    *status = MINIMUM_STEP;
                    return;
                }
            }
        }
        
        /* Calculate new gradient. */
        calcLLGradient(u_r, data, gamma, gradient);
        calcNCGradientIFFT(u_fft_r, u_fft_c, otf_mask_sqr, work1);
        vecfmaInplace(gradient, work1, alpha);        

        /* Convergence check. */
        if (converged(u_r, gradient)){
            *iterations = k+1;
            *status = SUCCESS;
            return;
        }
        
        /*
	 * Machine precision check.
	 *
	 * This is probably not an actual failure, we just ran out of digits. Reaching
	 * this state has a cost so we want to know if this is happening a lot.
	 */
        if (vecisEqual(u_r, u_p)){
            *iterations = k+1;
            *status = REACHED_MAXPRECISION;
            return;
        }
        
        /* L-BFGS calculation of new search direction. */
        ci = (k-1)%M;
        vecsub(s[ci], u_r, u_p);
        vecsub(y[ci], gradient, g_p);
        
        ys_c0 = vecdot(s[ci], y[ci]);
        ys[ci] = 1.0/ys_c0;
        yy = 1.0/vecdot(y[ci], y[ci]);
        
        vecncopy(srch_dir, gradient);
        bound = min(k, M);
        for(j=0; j<bound; j++){
	    ci = (k - j - 1)%M;
            a[ci] = vecdot(s[ci], srch_dir)*ys[ci];
            vecfmaInplace(srch_dir, y[ci], -a[ci]);
        }
        
        vecscaleInplace(srch_dir, ys_c0*yy);
        
        for(j=0; j<bound; j++){
	    ci = (k + j - bound)%M;
            beta = vecdot(y[ci], srch_dir)*ys[ci];
            vecfmaInplace(srch_dir, s[ci], (a[ci] - beta));
        }
        
        step = 1.0;
    }
    
    /* Reached maximum iterations. Minimization failed. */
    *iterations = MAXITERS;
    *status = REACHED_MAXITERS;
}
