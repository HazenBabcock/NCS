/*
 * CUDA kernel code for NCS.
 *
 * Hazen 08/19
 */

/*
 * Run NCS noise reduction on sub-regions.
 * 
 * Note: Any zero or negative values in the sub-regions should be
 *       set to a small positive value like 1.0.
 *
 * data_in - Sub-region data in e-.
 * g_gamma - Sub-region CMOS variance in units of e-^2.
 * otf_mask - 16 x 16 array containing the OTF mask.
 * data_out - Storage for noise corrected sub-regions.
 * iterations - Number of L-BFGS solver iterations.
 * status - Status of the solution (good, failed because of X).
 * alpha - NCS alpha term.
 */
__global__ void ncsReduceNoise(float4 *data_in,
                               float4 *g_gamma,
			       float4 *otf_mask,
			       float4 *data_out,
			       int *g_iterations,
			       int *g_status,
			       float alpha)
{
    int g_id = threadIdx.x;
    int offset = g_id*PSIZE;

    /* Variables. */
    int i;
    int iterations;
    int status;
    
    float4 data[PSIZE];
    float4 gamma[PSIZE];
    float4 otf_mask_sqr[PSIZE];
    float4 u_r[PSIZE]; 

    /* Initialization. */    
    for (i=0; i<PSIZE; i++){
        data[i] = data_in[i + offset];
        gamma[i] = g_gamma[i + offset];
        otf_mask_sqr[i] = otf_mask[i] * otf_mask[i];
        u_r[i] = data_in[i + offset];
    }

    /* Run NCS calculation. */
    ncsReduceNoiseSR(data, gamma, otf_mask_sqr, u_r, &iterations, &status, alpha);

    /* Save results. */
    for (i=0; i<PSIZE; i++){
    	data_out[i + offset] = u_r[i];
    }
    g_iterations[g_id] = iterations;
    g_status[g_id] = status;
}
