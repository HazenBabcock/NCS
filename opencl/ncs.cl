/*
 * OpenCL kernel code for NCS.
 *
 * Hazen 07/19
 */

/* Threshold for handling negative values in the fit. */
#define FITMIN 1.0e-6

/* The problem size is (16*16)/4 or 256/4. */
#define PSIZE 64

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

// 4 point complex FFT
void fft4(float4 x_r, float4 x_c, float4 *y_r, float4 *y_c)
{
    float t1_r = x_r.s0 + x_r.s2;
    float t1_c = x_c.s0 + x_c.s2;
    
    float t2_r = x_r.s0 - x_r.s2;
    float t2_c = x_c.s0 - x_c.s2;

    float t3_r = x_r.s1 + x_r.s3;
    float t3_c = x_c.s1 + x_c.s3;

    float t4_r = x_r.s1 - x_r.s3;
    float t4_c = x_c.s1 - x_c.s3;
 
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
    float t1_r = x_r.s0 + x_r.s2;
    float t1_c = x_c.s0 + x_c.s2;
    
    float t2_r = x_r.s0 - x_r.s2;
    float t2_c = x_c.s0 - x_c.s2;

    float t3_r = x_r.s1 + x_r.s3;
    float t3_c = x_c.s1 + x_c.s3;

    float t4_r = x_r.s1 - x_r.s3;
    float t4_c = x_c.s1 - x_c.s3;
 
    y_r[0].s0 = t1_r + t3_r;
    y_r[0].s1 = t2_r - t4_c;
    y_r[0].s2 = t1_r - t3_r;
    y_r[0].s3 = t2_r + t4_c;

    y_c[0].s0 = t1_c + t3_c;
    y_c[0].s1 = t2_c + t4_r;
    y_c[0].s2 = t1_c - t3_c;
    y_c[0].s3 = t2_c - t4_r;
    
    y_r[0] = y_r[0]*(float)0.25;
    y_c[0] = y_c[0]*(float)0.25;
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
    t_r = (float4)(x_r[0].s0, x_r[0].s2, x_r[1].s0, x_r[1].s2);
    t_c = (float4)(x_c[0].s0, x_c[0].s2, x_c[1].s0, x_c[1].s2);    
    fft4(t_r, t_c, &f41_r, &f41_c);

    t_r = (float4)(x_r[0].s1, x_r[0].s3, x_r[1].s1, x_r[1].s3);
    t_c = (float4)(x_c[0].s1, x_c[0].s3, x_c[1].s1, x_c[1].s3);
    fft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = (float4)(1.0, 7.07106781e-01, 0.0, -7.07106781e-01);
    float4 c1 = (float4)(0.0, 7.07106781e-01, 1.0,  7.07106781e-01);
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
    t_r = (float4)(x_r[0].s0, x_r[0].s2, x_r[1].s0, x_r[1].s2);
    t_c = (float4)(x_c[0].s0, x_c[0].s2, x_c[1].s0, x_c[1].s2);    
    ifft4(t_r, t_c, &f41_r, &f41_c);

    t_r = (float4)(x_r[0].s1, x_r[0].s3, x_r[1].s1, x_r[1].s3);
    t_c = (float4)(x_c[0].s1, x_c[0].s3, x_c[1].s1, x_c[1].s3);
    ifft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = (float4)(1.0,  7.07106781e-01,  0.0, -7.07106781e-01);
    float4 c1 = (float4)(0.0, -7.07106781e-01, -1.0, -7.07106781e-01);
    y_r[0] = f41_r + f42_r * r1 + f42_c * c1;
    y_c[0] = f41_c + f42_c * r1 - f42_r * c1;
    y_r[1] = f41_r - f42_r * r1 - f42_c * c1;
    y_c[1] = f41_c - f42_c * r1 + f42_r * c1;
    
    y_r[0] = y_r[0]*(float)0.5;
    y_c[0] = y_c[0]*(float)0.5;
    y_r[1] = y_r[1]*(float)0.5;
    y_c[1] = y_c[1]*(float)0.5;

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
    t_r[0] = (float4)(x_r[0].s0, x_r[0].s2, x_r[1].s0, x_r[1].s2);
    t_c[0] = (float4)(x_c[0].s0, x_c[0].s2, x_c[1].s0, x_c[1].s2);    
    t_r[1] = (float4)(x_r[2].s0, x_r[2].s2, x_r[3].s0, x_r[3].s2);
    t_c[1] = (float4)(x_c[2].s0, x_c[2].s2, x_c[3].s0, x_c[3].s2);    
    fft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = (float4)(x_r[0].s1, x_r[0].s3, x_r[1].s1, x_r[1].s3);
    t_c[0] = (float4)(x_c[0].s1, x_c[0].s3, x_c[1].s1, x_c[1].s3);
    t_r[1] = (float4)(x_r[2].s1, x_r[2].s3, x_r[3].s1, x_r[3].s3);
    t_c[1] = (float4)(x_c[2].s1, x_c[2].s3, x_c[3].s1, x_c[3].s3);
    fft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = (float4)(1.0, 9.23879533e-01, 7.07106781e-01, 3.82683432e-01);
    float4 c1 = (float4)(0.0, 3.82683432e-01, 7.07106781e-01, 9.23879533e-01);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = (float4)(0.0, -3.82683432e-01, -7.07106781e-01, -9.23879533e-01);
    c1 = (float4)(1.0,  9.23879533e-01,  7.07106781e-01,  3.82683432e-01);
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
    t_r[0] = (float4)(x_r[0].s0, x_r[0].s2, x_r[1].s0, x_r[1].s2);
    t_c[0] = (float4)(x_c[0].s0, x_c[0].s2, x_c[1].s0, x_c[1].s2);    
    t_r[1] = (float4)(x_r[2].s0, x_r[2].s2, x_r[3].s0, x_r[3].s2);
    t_c[1] = (float4)(x_c[2].s0, x_c[2].s2, x_c[3].s0, x_c[3].s2);    
    ifft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = (float4)(x_r[0].s1, x_r[0].s3, x_r[1].s1, x_r[1].s3);
    t_c[0] = (float4)(x_c[0].s1, x_c[0].s3, x_c[1].s1, x_c[1].s3);
    t_r[1] = (float4)(x_r[2].s1, x_r[2].s3, x_r[3].s1, x_r[3].s3);
    t_c[1] = (float4)(x_c[2].s1, x_c[2].s3, x_c[3].s1, x_c[3].s3);
    ifft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = (float4)(1.0,  9.23879533e-01,  7.07106781e-01,  3.82683432e-01);
    float4 c1 = (float4)(0.0, -3.82683432e-01, -7.07106781e-01, -9.23879533e-01);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = (float4)( 0.0, -3.82683432e-01, -7.07106781e-01, -9.23879533e-01);
    c1 = (float4)(-1.0, -9.23879533e-01, -7.07106781e-01, -3.82683432e-01);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;
    
    for(int i = 0; i<4; i++){
        y_r[i] = y_r[i]*(float)0.5;
        y_c[i] = y_c[i]*(float)0.5;    
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
            t1_r[j].s0 = y1_r[(4*j+0)*16+i];
            t1_r[j].s1 = y1_r[(4*j+1)*16+i];
            t1_r[j].s2 = y1_r[(4*j+2)*16+i];
            t1_r[j].s3 = y1_r[(4*j+3)*16+i];
            t1_c[j].s0 = y1_c[(4*j+0)*16+i];
            t1_c[j].s1 = y1_c[(4*j+1)*16+i];
            t1_c[j].s2 = y1_c[(4*j+2)*16+i];
            t1_c[j].s3 = y1_c[(4*j+3)*16+i];    
        }
        
        fft16(t1_r, t1_c, t1_r, t1_c);
        
        // Reverse conversion.
        for(int j=0; j<4; j++){
            y1_r[(4*j+0)*16+i] = t1_r[j].s0;
            y1_r[(4*j+1)*16+i] = t1_r[j].s1;
            y1_r[(4*j+2)*16+i] = t1_r[j].s2;
            y1_r[(4*j+3)*16+i] = t1_r[j].s3;
            y1_c[(4*j+0)*16+i] = t1_c[j].s0;
            y1_c[(4*j+1)*16+i] = t1_c[j].s1;
            y1_c[(4*j+2)*16+i] = t1_c[j].s2;
            y1_c[(4*j+3)*16+i] = t1_c[j].s3;            
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
            t1_r[j].s0 = x1_r[(4*j+0)*16+i];
            t1_r[j].s1 = x1_r[(4*j+1)*16+i];
            t1_r[j].s2 = x1_r[(4*j+2)*16+i];
            t1_r[j].s3 = x1_r[(4*j+3)*16+i];
            t1_c[j].s0 = x1_c[(4*j+0)*16+i];
            t1_c[j].s1 = x1_c[(4*j+1)*16+i];
            t1_c[j].s2 = x1_c[(4*j+2)*16+i];
            t1_c[j].s3 = x1_c[(4*j+3)*16+i];    
        }
        
        ifft16(t1_r, t1_c, t1_r, t1_c);
        
        // Reverse conversion.
        for(int j=0; j<4; j++){
            y1_r[(4*j+0)*16+i] = t1_r[j].s0;
            y1_r[(4*j+1)*16+i] = t1_r[j].s1;
            y1_r[(4*j+2)*16+i] = t1_r[j].s2;
            y1_r[(4*j+3)*16+i] = t1_r[j].s3;
            y1_c[(4*j+0)*16+i] = t1_c[j].s0;
            y1_c[(4*j+1)*16+i] = t1_c[j].s1;
            y1_c[(4*j+2)*16+i] = t1_c[j].s2;
            y1_c[(4*j+3)*16+i] = t1_c[j].s3;            
        }        
    }
    
    // Axis 1.
    for(int i=0; i<16; i++){
        ifft16(&(y_r[i*4]), &(y_c[i*4]), &(y_r[i*4]), &(y_c[i*4]));
    }     
}
