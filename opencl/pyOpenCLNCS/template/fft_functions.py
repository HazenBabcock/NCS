#!/usr/bin/env python
#
# Templates for the FFT functions.
#
# group_size - The number of workers in a group (this must be one of 16,32,64,128 or 256).
# item_size - The number of elements each worker is responsible for (this is a power of 2).
#
# Hazen 08/19
#

import jinja2


##
## FFT2
##
## Note: These are also responsible for creating all the variables that will be used.
##

# FFT2, 16 workers, fast axis.
fft2_gs16_fa_tpl = jinja2.Template("""
    int r_off = 16*lid;
    int i,j,k;

    {{real_type}} cf_r[16];
    {{real_type}} cf_c[16];
    {{real_type}} t_r[16];
    {{real_type}} t_c[16];

    i = 0;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    {{sync_fn}}

""")



# FFT2, 32 workers, fast axis.
fft2_gs32_fa_tpl = jinja2.Template("""
    int r_off = 8*(lid&30); // 0x00011110 
    int wn    = lid&1;      // 0x00000001
    int c_off, o_off;
    int i,j,k;

    {{real_type}} cf_r[16];
    {{real_type}} cf_c[16];
    {{real_type}} t_r[16];
    {{real_type}} t_c[16];

    i = 4*wn;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    {{sync_fn}}

""")


# FFT2, 64 workers, fast axis.
fft2_gs64_fa_tpl = jinja2.Template("""
    int r_off = 4*(lid&60); // 0x00111100 
    int wn    = lid&3;      // 0x00000011
    int c_off, o_off;
    int i,j,k;

    {{real_type}} cf_r[16];
    {{real_type}} cf_c[16];
    {{real_type}} t_r[16];
    {{real_type}} t_c[16];

    i = 2*wn;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    i += 1;
    j = r_off+i;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    {{sync_fn}}

""")


# FFT2, 128 workers, fast axis.
fft2_gs128_fa_tpl = jinja2.Template("""
    int r_off = 2*(lid&120); // 0x01111000 
    int wn    = lid&7;       // 0x00000111
    int c_off, o_off;
    int i,j,k;

    {{real_type}} cf_r[16];
    {{real_type}} cf_c[16];
    {{real_type}} t_r[16];
    {{real_type}} t_c[16];

    j = r_off+wn;
    {{a1}}[j]   = {{a3}}[j]+{{a3}}[j+8];
    {{a2}}[j]   = 0.0{{real_type[0]}};
    {{a1}}[j+8] = {{a3}}[j]-{{a3}}[j+8];
    {{a2}}[j+8] = 0.0{{real_type[0]}};

    {{sync_fn}}

""")


# FFT2, 256 workers, fast axis.
fft2_gs256_fa_tpl = jinja2.Template("""
    int r_off = lid&240; // 0x11110000 
    int wn    = lid&15;  // 0x00001111
    int c_off, o_off;
    int i,j,k;

    {{real_type}} cf_r[16];
    {{real_type}} cf_c[16];
    {{real_type}} t_r[16];
    {{real_type}} t_c[16];

    cf_r[0] =  1.0{{real_type[0]}}; 
    cf_r[1] = -1.0{{real_type[0]}};

    c_off = wn>>3;
    o_off = wn&7;
    j = r_off+o_off;
    {{a1}}[lid] = {{a3}}[j]+cf_r[c_off]*{{a3}}[j+8];
    {{a2}}[lid] = 0.0{{real_type[0]}};

    {{sync_fn}}

""")


##
## FFT4
## 

# FFT4, 16 workers, fast axis.
fft4_gs16_fa_tpl = jinja2.Template("""
    i = 0;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];

    i += 1;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];
   
    i += 1;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];

    i += 1;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];

    {{sync_fn}}
""")


# FFT4, 32 workers, fast axis.
fft4_gs32_fa_tpl = jinja2.Template("""
    i = 2*wn;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];

    i += 1;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[i+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];
   
    {{sync_fn}}
""")


# FFT4, 64 workers, fast axis.
fft4_gs64_fa_tpl = jinja2.Template("""
    i = wn;
    j = r_off+i;
    t_r[0] = {{a1}}[j] + {{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];
    t_r[2] = {{a1}}[j] - {{a1}}[j+4];
    t_r[3] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[j+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] =  {{a1}}[j+12];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+4]  = t_r[1];
    {{a1}}[j+8]  = t_r[2];
    {{a1}}[j+12] = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+4]  = t_c[1];
    {{a2}}[j+8]  = t_c[2];
    {{a2}}[j+12] = t_c[3];
   
    {{sync_fn}}
""")


# FFT4, 128 workers, fast axis.
fft4_gs128_fa_tpl = jinja2.Template("""
    cf_r[0] = 1.0{{real_type[0]}}; cf_r[1] = -1.0{{real_type[0]}};

    c_off = wn>>2;
    o_off = wn&3;
    j = r_off+o_off;
    t_r[0] = {{a1}}[j] + cf_r[c_off]*{{a1}}[j+4];
    t_r[1] = {{a1}}[j+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -cf_r[c_off]*{{a1}}[j+12];

    {{sync_fn}}

    j = r_off+8*c_off+o_off;
    {{a1}}[j]   = t_r[0];
    {{a1}}[j+4] = t_r[1];

    {{a2}}[j]   = t_c[0];
    {{a2}}[j+4] = t_c[1];

    {{sync_fn}}
""")


# FFT4, 256 workers, fast axis.
fft4_gs256_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  0.0{{real_type[0]}};
    cf_r[2] = -1.0{{real_type[0]}};
    cf_r[3] =  0.0{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -1.0{{real_type[0]}};
    cf_c[2] =  0.0{{real_type[0]}};
    cf_c[3] =  1.0{{real_type[0]}};

    c_off = wn>>2;
    o_off = wn&3;
    j = r_off+o_off;
    k = j + 8*(c_off&1);

    t_r[0] = {{a1}}[k] + cf_r[c_off]*{{a1}}[k+4];
    t_c[0] = cf_c[c_off]*{{a1}}[j+12];

    {{sync_fn}}

    {{a1}}[lid] = t_r[0];
    {{a2}}[lid] = t_c[0];
   
    {{sync_fn}}
""")


##
## FFT8
## 

# FFT8, 16 workers, fast axis.
fft8_gs16_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  7.07106781e-01{{real_type[0]}};
    cf_r[2] =  0.0{{real_type[0]}};
    cf_r[3] =  -7.07106781e-01{{real_type[0]}};
    cf_r[4] =  -1.0{{real_type[0]}};
    cf_r[5] =  -7.07106781e-01{{real_type[0]}};
    cf_r[6] =  0.0{{real_type[0]}};
    cf_r[7] =  7.07106781e-01{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -7.07106781e-01{{real_type[0]}};
    cf_c[2] = -1.0{{real_type[0]}};
    cf_c[3] = -7.07106781e-01{{real_type[0]}};
    cf_c[4] =  0.0{{real_type[0]}};
    cf_c[5] =  7.07106781e-01{{real_type[0]}};
    cf_c[6] =  1.0{{real_type[0]}};
    cf_c[7] =  7.07106781e-01{{real_type[0]}};

    j = r_off;
    t_r[0] = {{a1}}[j]    + cf_r[0]*{{a1}}[j+2]  - cf_c[0]*{{a2}}[j+2];
    t_r[1] = {{a1}}[j+4]  + cf_r[1]*{{a1}}[j+6]  - cf_c[1]*{{a2}}[j+6];
    t_r[2] = {{a1}}[j+8]  + cf_r[2]*{{a1}}[j+10] - cf_c[2]*{{a2}}[j+10];
    t_r[3] = {{a1}}[j+12] + cf_r[3]*{{a1}}[j+14] - cf_c[3]*{{a2}}[j+14];
    t_r[4] = {{a1}}[j]    + cf_r[4]*{{a1}}[j+2]  - cf_c[4]*{{a2}}[j+2];
    t_r[5] = {{a1}}[j+4]  + cf_r[5]*{{a1}}[j+6]  - cf_c[5]*{{a2}}[j+6];
    t_r[6] = {{a1}}[j+8]  + cf_r[6]*{{a1}}[j+10] - cf_c[6]*{{a2}}[j+10];
    t_r[7] = {{a1}}[j+12] + cf_r[7]*{{a1}}[j+14] - cf_c[7]*{{a2}}[j+14];

    t_c[0] = {{a2}}[j]    + cf_r[0]*{{a2}}[j+2]  + cf_c[0]*{{a1}}[j+2]; 
    t_c[1] = {{a2}}[j+4]  + cf_r[1]*{{a2}}[j+6]  + cf_c[1]*{{a1}}[j+6];
    t_c[2] = {{a2}}[j+8]  + cf_r[2]*{{a2}}[j+10] + cf_c[2]*{{a1}}[j+10];
    t_c[3] = {{a2}}[j+12] + cf_r[3]*{{a2}}[j+14] + cf_c[3]*{{a1}}[j+14];
    t_c[4] = {{a2}}[j]    + cf_r[4]*{{a2}}[j+2]  + cf_c[4]*{{a1}}[j+2];
    t_c[5] = {{a2}}[j+4]  + cf_r[5]*{{a2}}[j+6]  + cf_c[5]*{{a1}}[j+6];
    t_c[6] = {{a2}}[j+8]  + cf_r[6]*{{a2}}[j+10] + cf_c[6]*{{a1}}[j+10];
    t_c[7] = {{a2}}[j+12] + cf_r[7]*{{a2}}[j+14] + cf_c[7]*{{a1}}[j+14];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+2]  = t_r[1];
    {{a1}}[j+4]  = t_r[2];
    {{a1}}[j+6]  = t_r[3];
    {{a1}}[j+8]  = t_r[4];
    {{a1}}[j+10] = t_r[5];
    {{a1}}[j+12] = t_r[6];
    {{a1}}[j+14] = t_r[7];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+2]  = t_c[1];
    {{a2}}[j+4]  = t_c[2];
    {{a2}}[j+6]  = t_c[3];
    {{a2}}[j+8]  = t_c[4];
    {{a2}}[j+10] = t_c[5];
    {{a2}}[j+12] = t_c[6];
    {{a2}}[j+14] = t_c[7];

    j = r_off+1;
    t_r[0] = {{a1}}[j]    + cf_r[0]*{{a1}}[j+2]  - cf_c[0]*{{a2}}[j+2];
    t_r[1] = {{a1}}[j+4]  + cf_r[1]*{{a1}}[j+6]  - cf_c[1]*{{a2}}[j+6];
    t_r[2] = {{a1}}[j+8]  + cf_r[2]*{{a1}}[j+10] - cf_c[2]*{{a2}}[j+10];
    t_r[3] = {{a1}}[j+12] + cf_r[3]*{{a1}}[j+14] - cf_c[3]*{{a2}}[j+14];
    t_r[4] = {{a1}}[j]    + cf_r[4]*{{a1}}[j+2]  - cf_c[4]*{{a2}}[j+2];
    t_r[5] = {{a1}}[j+4]  + cf_r[5]*{{a1}}[j+6]  - cf_c[5]*{{a2}}[j+6];
    t_r[6] = {{a1}}[j+8]  + cf_r[6]*{{a1}}[j+10] - cf_c[6]*{{a2}}[j+10];
    t_r[7] = {{a1}}[j+12] + cf_r[7]*{{a1}}[j+14] - cf_c[7]*{{a2}}[j+14];

    t_c[0] = {{a2}}[j]    + cf_r[0]*{{a2}}[j+2]  + cf_c[0]*{{a1}}[j+2]; 
    t_c[1] = {{a2}}[j+4]  + cf_r[1]*{{a2}}[j+6]  + cf_c[1]*{{a1}}[j+6];
    t_c[2] = {{a2}}[j+8]  + cf_r[2]*{{a2}}[j+10] + cf_c[2]*{{a1}}[j+10];
    t_c[3] = {{a2}}[j+12] + cf_r[3]*{{a2}}[j+14] + cf_c[3]*{{a1}}[j+14];
    t_c[4] = {{a2}}[j]    + cf_r[4]*{{a2}}[j+2]  + cf_c[4]*{{a1}}[j+2];
    t_c[5] = {{a2}}[j+4]  + cf_r[5]*{{a2}}[j+6]  + cf_c[5]*{{a1}}[j+6];
    t_c[6] = {{a2}}[j+8]  + cf_r[6]*{{a2}}[j+10] + cf_c[6]*{{a1}}[j+10];
    t_c[7] = {{a2}}[j+12] + cf_r[7]*{{a2}}[j+14] + cf_c[7]*{{a1}}[j+14];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+2]  = t_r[1];
    {{a1}}[j+4]  = t_r[2];
    {{a1}}[j+6]  = t_r[3];
    {{a1}}[j+8]  = t_r[4];
    {{a1}}[j+10] = t_r[5];
    {{a1}}[j+12] = t_r[6];
    {{a1}}[j+14] = t_r[7];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+2]  = t_c[1];
    {{a2}}[j+4]  = t_c[2];
    {{a2}}[j+6]  = t_c[3];
    {{a2}}[j+8]  = t_c[4];
    {{a2}}[j+10] = t_c[5];
    {{a2}}[j+12] = t_c[6];
    {{a2}}[j+14] = t_c[7];

    {{sync_fn}}

""")


# FFT8, 32 workers, fast axis.
fft8_gs32_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  7.07106781e-01{{real_type[0]}};
    cf_r[2] =  0.0{{real_type[0]}};
    cf_r[3] =  -7.07106781e-01{{real_type[0]}};
    cf_r[4] =  -1.0{{real_type[0]}};
    cf_r[5] =  -7.07106781e-01{{real_type[0]}};
    cf_r[6] =  0.0{{real_type[0]}};
    cf_r[7] =  7.07106781e-01{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -7.07106781e-01{{real_type[0]}};
    cf_c[2] = -1.0{{real_type[0]}};
    cf_c[3] = -7.07106781e-01{{real_type[0]}};
    cf_c[4] =  0.0{{real_type[0]}};
    cf_c[5] =  7.07106781e-01{{real_type[0]}};
    cf_c[6] =  1.0{{real_type[0]}};
    cf_c[7] =  7.07106781e-01{{real_type[0]}};

    j = r_off+wn;
    t_r[0] = {{a1}}[j]    + cf_r[0]*{{a1}}[j+2]  - cf_c[0]*{{a2}}[j+2];
    t_r[1] = {{a1}}[j+4]  + cf_r[1]*{{a1}}[j+6]  - cf_c[1]*{{a2}}[j+6];
    t_r[2] = {{a1}}[j+8]  + cf_r[2]*{{a1}}[j+10] - cf_c[2]*{{a2}}[j+10];
    t_r[3] = {{a1}}[j+12] + cf_r[3]*{{a1}}[j+14] - cf_c[3]*{{a2}}[j+14];
    t_r[4] = {{a1}}[j]    + cf_r[4]*{{a1}}[j+2]  - cf_c[4]*{{a2}}[j+2];
    t_r[5] = {{a1}}[j+4]  + cf_r[5]*{{a1}}[j+6]  - cf_c[5]*{{a2}}[j+6];
    t_r[6] = {{a1}}[j+8]  + cf_r[6]*{{a1}}[j+10] - cf_c[6]*{{a2}}[j+10];
    t_r[7] = {{a1}}[j+12] + cf_r[7]*{{a1}}[j+14] - cf_c[7]*{{a2}}[j+14];

    t_c[0] = {{a2}}[j]    + cf_r[0]*{{a2}}[j+2]  + cf_c[0]*{{a1}}[j+2]; 
    t_c[1] = {{a2}}[j+4]  + cf_r[1]*{{a2}}[j+6]  + cf_c[1]*{{a1}}[j+6];
    t_c[2] = {{a2}}[j+8]  + cf_r[2]*{{a2}}[j+10] + cf_c[2]*{{a1}}[j+10];
    t_c[3] = {{a2}}[j+12] + cf_r[3]*{{a2}}[j+14] + cf_c[3]*{{a1}}[j+14];
    t_c[4] = {{a2}}[j]    + cf_r[4]*{{a2}}[j+2]  + cf_c[4]*{{a1}}[j+2];
    t_c[5] = {{a2}}[j+4]  + cf_r[5]*{{a2}}[j+6]  + cf_c[5]*{{a1}}[j+6];
    t_c[6] = {{a2}}[j+8]  + cf_r[6]*{{a2}}[j+10] + cf_c[6]*{{a1}}[j+10];
    t_c[7] = {{a2}}[j+12] + cf_r[7]*{{a2}}[j+14] + cf_c[7]*{{a1}}[j+14];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+2]  = t_r[1];
    {{a1}}[j+4]  = t_r[2];
    {{a1}}[j+6]  = t_r[3];
    {{a1}}[j+8]  = t_r[4];
    {{a1}}[j+10] = t_r[5];
    {{a1}}[j+12] = t_r[6];
    {{a1}}[j+14] = t_r[7];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+2]  = t_c[1];
    {{a2}}[j+4]  = t_c[2];
    {{a2}}[j+6]  = t_c[3];
    {{a2}}[j+8]  = t_c[4];
    {{a2}}[j+10] = t_c[5];
    {{a2}}[j+12] = t_c[6];
    {{a2}}[j+14] = t_c[7];

    {{sync_fn}}
""")


# FFT8, 64 workers, fast axis.
fft8_gs64_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  7.07106781e-01{{real_type[0]}};
    cf_r[2] =  0.0{{real_type[0]}};
    cf_r[3] =  -7.07106781e-01{{real_type[0]}};
    cf_r[4] =  -1.0{{real_type[0]}};
    cf_r[5] =  -7.07106781e-01{{real_type[0]}};
    cf_r[6] =  0.0{{real_type[0]}};
    cf_r[7] =  7.07106781e-01{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -7.07106781e-01{{real_type[0]}};
    cf_c[2] = -1.0{{real_type[0]}};
    cf_c[3] = -7.07106781e-01{{real_type[0]}};
    cf_c[4] =  0.0{{real_type[0]}};
    cf_c[5] =  7.07106781e-01{{real_type[0]}};
    cf_c[6] =  1.0{{real_type[0]}};
    cf_c[7] =  7.07106781e-01{{real_type[0]}};
 
    c_off = 4*(wn&1);
    o_off = wn>>1;
    j = r_off+o_off;

    t_r[0] = {{a1}}[j]    + cf_r[c_off]  *{{a1}}[j+2]  - cf_c[c_off]  *{{a2}}[j+2];
    t_r[1] = {{a1}}[j+4]  + cf_r[c_off+1]*{{a1}}[j+6]  - cf_c[c_off+1]*{{a2}}[j+6];
    t_r[2] = {{a1}}[j+8]  + cf_r[c_off+2]*{{a1}}[j+10] - cf_c[c_off+2]*{{a2}}[j+10];
    t_r[3] = {{a1}}[j+12] + cf_r[c_off+3]*{{a1}}[j+14] - cf_c[c_off+3]*{{a2}}[j+14];

    t_c[0] = {{a2}}[j]    + cf_r[c_off]  *{{a2}}[j+2]  + cf_c[c_off]  *{{a1}}[j+2]; 
    t_c[1] = {{a2}}[j+4]  + cf_r[c_off+1]*{{a2}}[j+6]  + cf_c[c_off+1]*{{a1}}[j+6];
    t_c[2] = {{a2}}[j+8]  + cf_r[c_off+2]*{{a2}}[j+10] + cf_c[c_off+2]*{{a1}}[j+10];
    t_c[3] = {{a2}}[j+12] + cf_r[c_off+3]*{{a2}}[j+14] + cf_c[c_off+3]*{{a1}}[j+14];

    {{sync_fn}}

    j = r_off + 2*c_off + o_off;
    {{a1}}[j]    = t_r[0];
    {{a1}}[j+2]  = t_r[1];
    {{a1}}[j+4]  = t_r[2];
    {{a1}}[j+6]  = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+2]  = t_c[1];
    {{a2}}[j+4]  = t_c[2];
    {{a2}}[j+6]  = t_c[3];

    {{sync_fn}}
""")


# FFT8, 128 workers, fast axis.
fft8_gs128_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  7.07106781e-01{{real_type[0]}};
    cf_r[2] =  0.0{{real_type[0]}};
    cf_r[3] =  -7.07106781e-01{{real_type[0]}};
    cf_r[4] =  -1.0{{real_type[0]}};
    cf_r[5] =  -7.07106781e-01{{real_type[0]}};
    cf_r[6] =  0.0{{real_type[0]}};
    cf_r[7] =  7.07106781e-01{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -7.07106781e-01{{real_type[0]}};
    cf_c[2] = -1.0{{real_type[0]}};
    cf_c[3] = -7.07106781e-01{{real_type[0]}};
    cf_c[4] =  0.0{{real_type[0]}};
    cf_c[5] =  7.07106781e-01{{real_type[0]}};
    cf_c[6] =  1.0{{real_type[0]}};
    cf_c[7] =  7.07106781e-01{{real_type[0]}};
 
    c_off = 2*(wn&3);
    o_off = wn>>2;
    j = r_off + o_off;
    k = j + 8*(wn&1);

    t_r[0] = {{a1}}[k]   + cf_r[c_off]  *{{a1}}[k+2] - cf_c[c_off]  *{{a2}}[k+2];
    t_r[1] = {{a1}}[k+4] + cf_r[c_off+1]*{{a1}}[k+6] - cf_c[c_off+1]*{{a2}}[k+6];

    t_c[0] = {{a2}}[k]   + cf_r[c_off]  *{{a2}}[k+2] + cf_c[c_off]  *{{a1}}[k+2]; 
    t_c[1] = {{a2}}[k+4] + cf_r[c_off+1]*{{a2}}[k+6] + cf_c[c_off+1]*{{a1}}[k+6];

    {{sync_fn}}

    j = r_off + 2*c_off + o_off;
    {{a1}}[j]    = t_r[0];
    {{a1}}[j+2]  = t_r[1];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+2]  = t_c[1];

    {{sync_fn}}

""")


# FFT8, 256 workers, fast axis.
fft8_gs256_fa_tpl = jinja2.Template("""
    cf_r[0] =  1.0{{real_type[0]}};
    cf_r[1] =  7.07106781e-01{{real_type[0]}};
    cf_r[2] =  0.0{{real_type[0]}};
    cf_r[3] =  -7.07106781e-01{{real_type[0]}};
    cf_r[4] =  -1.0{{real_type[0]}};
    cf_r[5] =  -7.07106781e-01{{real_type[0]}};
    cf_r[6] =  0.0{{real_type[0]}};
    cf_r[7] =  7.07106781e-01{{real_type[0]}};

    cf_c[0] =  0.0{{real_type[0]}};
    cf_c[1] = -7.07106781e-01{{real_type[0]}};
    cf_c[2] = -1.0{{real_type[0]}};
    cf_c[3] = -7.07106781e-01{{real_type[0]}};
    cf_c[4] =  0.0{{real_type[0]}};
    cf_c[5] =  7.07106781e-01{{real_type[0]}};
    cf_c[6] =  1.0{{real_type[0]}};
    cf_c[7] =  7.07106781e-01{{real_type[0]}};
 
    c_off = wn&7;
    o_off = wn>>3;
    j = r_off + o_off;
    k = j + 4*(wn&3);

    t_r[0] = {{a1}}[k] + cf_r[c_off]*{{a1}}[k+2] - cf_c[c_off]*{{a2}}[k+2];
    t_c[0] = {{a2}}[k] + cf_r[c_off]*{{a2}}[k+2] + cf_c[c_off]*{{a1}}[k+2]; 

    {{sync_fn}}

    j = r_off + 2*c_off + o_off;
    {{a1}}[j] = t_r[0];
    {{a2}}[j] = t_c[0];

    {{sync_fn}}

""")


##
## FFT16
## 

# FFT16, 16 workers, fast axis.
fft16_gs16_fa_tpl = jinja2.Template("""
    cf_r[0]  =  1.0{{real_type[0]}};
    cf_r[1]  =  9.23879533e-01{{real_type[0]}};
    cf_r[2]  =  7.07106781e-01{{real_type[0]}};
    cf_r[3]  =  3.82683432e-01{{real_type[0]}};
    cf_r[4]  =  0.0{{real_type[0]}};
    cf_r[5]  = -3.82683432e-01{{real_type[0]}};
    cf_r[6]  = -7.07106781e-01{{real_type[0]}};
    cf_r[7]  = -9.23879533e-01{{real_type[0]}};
    cf_r[8]  = -1.0{{real_type[0]}};
    cf_r[9]  = -9.23879533e-01{{real_type[0]}};
    cf_r[10] = -7.07106781e-01{{real_type[0]}};
    cf_r[11] = -3.82683432e-01{{real_type[0]}};
    cf_r[12] =  0.0{{real_type[0]}};
    cf_r[13] =  3.82683432e-01{{real_type[0]}};
    cf_r[14] =  7.07106781e-01{{real_type[0]}};
    cf_r[15] =  9.23879533e-01{{real_type[0]}};

    cf_c[0]  =  0.0{{real_type[0]}};
    cf_c[1]  = -3.82683432e-01{{real_type[0]}};
    cf_c[2]  = -7.07106781e-01{{real_type[0]}};
    cf_c[3]  = -9.23879533e-01{{real_type[0]}};
    cf_c[4]  = -1.0{{real_type[0]}};
    cf_c[5]  = -9.23879533e-01{{real_type[0]}};
    cf_c[6]  = -7.07106781e-01{{real_type[0]}};
    cf_c[7]  = -3.82683432e-01{{real_type[0]}};
    cf_c[8]  =  0.0{{real_type[0]}};
    cf_c[9]  =  3.82683432e-01{{real_type[0]}};
    cf_c[10] =  7.07106781e-01{{real_type[0]}};
    cf_c[11] =  9.23879533e-01{{real_type[0]}};
    cf_c[12] =  1.0{{real_type[0]}};
    cf_c[13] =  9.23879533e-01{{real_type[0]}};
    cf_c[14] =  7.07106781e-01{{real_type[0]}};
    cf_c[15] =  3.82683432e-01{{real_type[0]}};

    j = r_off;
    t_r[0]  = {{a1}}[j]    + cf_r[0]* {{a1}}[j+1]  - cf_c[0]* {{a2}}[j+1];
    t_r[1]  = {{a1}}[j+2]  + cf_r[1]* {{a1}}[j+3]  - cf_c[1]* {{a2}}[j+3];
    t_r[2]  = {{a1}}[j+4]  + cf_r[2]* {{a1}}[j+5]  - cf_c[2]* {{a2}}[j+5];
    t_r[3]  = {{a1}}[j+6]  + cf_r[3]* {{a1}}[j+7]  - cf_c[3]* {{a2}}[j+7];
    t_r[4]  = {{a1}}[j+8]  + cf_r[4]* {{a1}}[j+9]  - cf_c[4]* {{a2}}[j+9];
    t_r[5]  = {{a1}}[j+10] + cf_r[5]* {{a1}}[j+11] - cf_c[5]* {{a2}}[j+11];
    t_r[6]  = {{a1}}[j+12] + cf_r[6]* {{a1}}[j+13] - cf_c[6]* {{a2}}[j+13];
    t_r[7]  = {{a1}}[j+14] + cf_r[7]* {{a1}}[j+15] - cf_c[7]* {{a2}}[j+15];
    t_r[8]  = {{a1}}[j]    + cf_r[8]* {{a1}}[j+1]  - cf_c[8]* {{a2}}[j+1];
    t_r[9]  = {{a1}}[j+2]  + cf_r[9]* {{a1}}[j+3]  - cf_c[9]* {{a2}}[j+3];
    t_r[10] = {{a1}}[j+4]  + cf_r[10]*{{a1}}[j+5]  - cf_c[10]*{{a2}}[j+5];
    t_r[11] = {{a1}}[j+6]  + cf_r[11]*{{a1}}[j+7]  - cf_c[11]*{{a2}}[j+7];
    t_r[12] = {{a1}}[j+8]  + cf_r[12]*{{a1}}[j+9]  - cf_c[12]*{{a2}}[j+9];
    t_r[13] = {{a1}}[j+10] + cf_r[13]*{{a1}}[j+11] - cf_c[13]*{{a2}}[j+11];
    t_r[14] = {{a1}}[j+12] + cf_r[14]*{{a1}}[j+13] - cf_c[14]*{{a2}}[j+13];
    t_r[15] = {{a1}}[j+14] + cf_r[15]*{{a1}}[j+15] - cf_c[15]*{{a2}}[j+15];

    t_c[0]  = {{a2}}[j]    + cf_r[0]* {{a2}}[j+1]  + cf_c[0]* {{a1}}[j+1]; 
    t_c[1]  = {{a2}}[j+2]  + cf_r[1]* {{a2}}[j+3]  + cf_c[1]* {{a1}}[j+3];
    t_c[2]  = {{a2}}[j+4]  + cf_r[2]* {{a2}}[j+5]  + cf_c[2]* {{a1}}[j+5];
    t_c[3]  = {{a2}}[j+6]  + cf_r[3]* {{a2}}[j+7]  + cf_c[3]* {{a1}}[j+7];
    t_c[4]  = {{a2}}[j+8]  + cf_r[4]* {{a2}}[j+9]  + cf_c[4]* {{a1}}[j+9];
    t_c[5]  = {{a2}}[j+10] + cf_r[5]* {{a2}}[j+11] + cf_c[5]* {{a1}}[j+11];
    t_c[6]  = {{a2}}[j+12] + cf_r[6]* {{a2}}[j+13] + cf_c[6]* {{a1}}[j+13];
    t_c[7]  = {{a2}}[j+14] + cf_r[7]* {{a2}}[j+15] + cf_c[7]* {{a1}}[j+15];
    t_c[8]  = {{a2}}[j]    + cf_r[8]* {{a2}}[j+1]  + cf_c[8]* {{a1}}[j+1]; 
    t_c[9]  = {{a2}}[j+2]  + cf_r[9]* {{a2}}[j+3]  + cf_c[9]* {{a1}}[j+3];
    t_c[10] = {{a2}}[j+4]  + cf_r[10]*{{a2}}[j+5]  + cf_c[10]*{{a1}}[j+5];
    t_c[11] = {{a2}}[j+6]  + cf_r[11]*{{a2}}[j+7]  + cf_c[11]*{{a1}}[j+7];
    t_c[12] = {{a2}}[j+8]  + cf_r[12]*{{a2}}[j+9]  + cf_c[12]*{{a1}}[j+9];
    t_c[13] = {{a2}}[j+10] + cf_r[13]*{{a2}}[j+11] + cf_c[13]*{{a1}}[j+11];
    t_c[14] = {{a2}}[j+12] + cf_r[14]*{{a2}}[j+13] + cf_c[14]*{{a1}}[j+13];
    t_c[15] = {{a2}}[j+14] + cf_r[15]*{{a2}}[j+15] + cf_c[15]*{{a1}}[j+15];

    {{a1}}[j]    = t_r[0];
    {{a1}}[j+1]  = t_r[1];
    {{a1}}[j+2]  = t_r[2];
    {{a1}}[j+3]  = t_r[3];
    {{a1}}[j+4]  = t_r[4];
    {{a1}}[j+5]  = t_r[5];
    {{a1}}[j+6]  = t_r[6];
    {{a1}}[j+7]  = t_r[7];
    {{a1}}[j+8]  = t_r[8];
    {{a1}}[j+9]  = t_r[9];
    {{a1}}[j+10] = t_r[10];
    {{a1}}[j+11] = t_r[11];
    {{a1}}[j+12] = t_r[12];
    {{a1}}[j+13] = t_r[13];
    {{a1}}[j+14] = t_r[14];
    {{a1}}[j+15] = t_r[15];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+1]  = t_c[1];
    {{a2}}[j+2]  = t_c[2];
    {{a2}}[j+3]  = t_c[3];
    {{a2}}[j+4]  = t_c[4];
    {{a2}}[j+5]  = t_c[5];
    {{a2}}[j+6]  = t_c[6];
    {{a2}}[j+7]  = t_c[7];
    {{a2}}[j+8]  = t_c[8];
    {{a2}}[j+9]  = t_c[9];
    {{a2}}[j+10] = t_c[10];
    {{a2}}[j+11] = t_c[11];
    {{a2}}[j+12] = t_c[12];
    {{a2}}[j+13] = t_c[13];
    {{a2}}[j+14] = t_c[14];
    {{a2}}[j+15] = t_c[15];

    {{sync_fn}}

""")


# FFT16, 32 workers, fast axis.
fft16_gs32_fa_tpl = jinja2.Template("""
    cf_r[0]  =  1.0{{real_type[0]}};
    cf_r[1]  =  9.23879533e-01{{real_type[0]}};
    cf_r[2]  =  7.07106781e-01{{real_type[0]}};
    cf_r[3]  =  3.82683432e-01{{real_type[0]}};
    cf_r[4]  =  0.0{{real_type[0]}};
    cf_r[5]  = -3.82683432e-01{{real_type[0]}};
    cf_r[6]  = -7.07106781e-01{{real_type[0]}};
    cf_r[7]  = -9.23879533e-01{{real_type[0]}};
    cf_r[8]  = -1.0{{real_type[0]}};
    cf_r[9]  = -9.23879533e-01{{real_type[0]}};
    cf_r[10] = -7.07106781e-01{{real_type[0]}};
    cf_r[11] = -3.82683432e-01{{real_type[0]}};
    cf_r[12] =  0.0{{real_type[0]}};
    cf_r[13] =  3.82683432e-01{{real_type[0]}};
    cf_r[14] =  7.07106781e-01{{real_type[0]}};
    cf_r[15] =  9.23879533e-01{{real_type[0]}};

    cf_c[0]  =  0.0{{real_type[0]}};
    cf_c[1]  = -3.82683432e-01{{real_type[0]}};
    cf_c[2]  = -7.07106781e-01{{real_type[0]}};
    cf_c[3]  = -9.23879533e-01{{real_type[0]}};
    cf_c[4]  = -1.0{{real_type[0]}};
    cf_c[5]  = -9.23879533e-01{{real_type[0]}};
    cf_c[6]  = -7.07106781e-01{{real_type[0]}};
    cf_c[7]  = -3.82683432e-01{{real_type[0]}};
    cf_c[8]  =  0.0{{real_type[0]}};
    cf_c[9]  =  3.82683432e-01{{real_type[0]}};
    cf_c[10] =  7.07106781e-01{{real_type[0]}};
    cf_c[11] =  9.23879533e-01{{real_type[0]}};
    cf_c[12] =  1.0{{real_type[0]}};
    cf_c[13] =  9.23879533e-01{{real_type[0]}};
    cf_c[14] =  7.07106781e-01{{real_type[0]}};
    cf_c[15] =  3.82683432e-01{{real_type[0]}};

    c_off = 8*wn;
    j = r_off;

    t_r[0]  = {{a1}}[j]    + cf_r[c_off]  * {{a1}}[j+1]  - cf_c[c_off]  * {{a2}}[j+1];
    t_r[1]  = {{a1}}[j+2]  + cf_r[c_off+1]* {{a1}}[j+3]  - cf_c[c_off+1]* {{a2}}[j+3];
    t_r[2]  = {{a1}}[j+4]  + cf_r[c_off+2]* {{a1}}[j+5]  - cf_c[c_off+2]* {{a2}}[j+5];
    t_r[3]  = {{a1}}[j+6]  + cf_r[c_off+3]* {{a1}}[j+7]  - cf_c[c_off+3]* {{a2}}[j+7];
    t_r[4]  = {{a1}}[j+8]  + cf_r[c_off+4]* {{a1}}[j+9]  - cf_c[c_off+4]* {{a2}}[j+9];
    t_r[5]  = {{a1}}[j+10] + cf_r[c_off+5]* {{a1}}[j+11] - cf_c[c_off+5]* {{a2}}[j+11];
    t_r[6]  = {{a1}}[j+12] + cf_r[c_off+6]* {{a1}}[j+13] - cf_c[c_off+6]* {{a2}}[j+13];
    t_r[7]  = {{a1}}[j+14] + cf_r[c_off+7]* {{a1}}[j+15] - cf_c[c_off+7]* {{a2}}[j+15];

    t_c[0]  = {{a2}}[j]    + cf_r[c_off]  * {{a2}}[j+1]  + cf_c[c_off]  * {{a1}}[j+1]; 
    t_c[1]  = {{a2}}[j+2]  + cf_r[c_off+1]* {{a2}}[j+3]  + cf_c[c_off+1]* {{a1}}[j+3];
    t_c[2]  = {{a2}}[j+4]  + cf_r[c_off+2]* {{a2}}[j+5]  + cf_c[c_off+2]* {{a1}}[j+5];
    t_c[3]  = {{a2}}[j+6]  + cf_r[c_off+3]* {{a2}}[j+7]  + cf_c[c_off+3]* {{a1}}[j+7];
    t_c[4]  = {{a2}}[j+8]  + cf_r[c_off+4]* {{a2}}[j+9]  + cf_c[c_off+4]* {{a1}}[j+9];
    t_c[5]  = {{a2}}[j+10] + cf_r[c_off+5]* {{a2}}[j+11] + cf_c[c_off+5]* {{a1}}[j+11];
    t_c[6]  = {{a2}}[j+12] + cf_r[c_off+6]* {{a2}}[j+13] + cf_c[c_off+6]* {{a1}}[j+13];
    t_c[7]  = {{a2}}[j+14] + cf_r[c_off+7]* {{a2}}[j+15] + cf_c[c_off+7]* {{a1}}[j+15];

    {{sync_fn}}

    j = r_off + c_off;
    {{a1}}[j]    = t_r[0];
    {{a1}}[j+1]  = t_r[1];
    {{a1}}[j+2]  = t_r[2];
    {{a1}}[j+3]  = t_r[3];
    {{a1}}[j+4]  = t_r[4];
    {{a1}}[j+5]  = t_r[5];
    {{a1}}[j+6]  = t_r[6];
    {{a1}}[j+7]  = t_r[7];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+1]  = t_c[1];
    {{a2}}[j+2]  = t_c[2];
    {{a2}}[j+3]  = t_c[3];
    {{a2}}[j+4]  = t_c[4];
    {{a2}}[j+5]  = t_c[5];
    {{a2}}[j+6]  = t_c[6];
    {{a2}}[j+7]  = t_c[7];

    {{sync_fn}}
""")


# FFT16, 64 workers, fast axis.
fft16_gs64_fa_tpl = jinja2.Template("""
    cf_r[0]  =  1.0{{real_type[0]}};
    cf_r[1]  =  9.23879533e-01{{real_type[0]}};
    cf_r[2]  =  7.07106781e-01{{real_type[0]}};
    cf_r[3]  =  3.82683432e-01{{real_type[0]}};
    cf_r[4]  =  0.0{{real_type[0]}};
    cf_r[5]  = -3.82683432e-01{{real_type[0]}};
    cf_r[6]  = -7.07106781e-01{{real_type[0]}};
    cf_r[7]  = -9.23879533e-01{{real_type[0]}};
    cf_r[8]  = -1.0{{real_type[0]}};
    cf_r[9]  = -9.23879533e-01{{real_type[0]}};
    cf_r[10] = -7.07106781e-01{{real_type[0]}};
    cf_r[11] = -3.82683432e-01{{real_type[0]}};
    cf_r[12] =  0.0{{real_type[0]}};
    cf_r[13] =  3.82683432e-01{{real_type[0]}};
    cf_r[14] =  7.07106781e-01{{real_type[0]}};
    cf_r[15] =  9.23879533e-01{{real_type[0]}};

    cf_c[0]  =  0.0{{real_type[0]}};
    cf_c[1]  = -3.82683432e-01{{real_type[0]}};
    cf_c[2]  = -7.07106781e-01{{real_type[0]}};
    cf_c[3]  = -9.23879533e-01{{real_type[0]}};
    cf_c[4]  = -1.0{{real_type[0]}};
    cf_c[5]  = -9.23879533e-01{{real_type[0]}};
    cf_c[6]  = -7.07106781e-01{{real_type[0]}};
    cf_c[7]  = -3.82683432e-01{{real_type[0]}};
    cf_c[8]  =  0.0{{real_type[0]}};
    cf_c[9]  =  3.82683432e-01{{real_type[0]}};
    cf_c[10] =  7.07106781e-01{{real_type[0]}};
    cf_c[11] =  9.23879533e-01{{real_type[0]}};
    cf_c[12] =  1.0{{real_type[0]}};
    cf_c[13] =  9.23879533e-01{{real_type[0]}};
    cf_c[14] =  7.07106781e-01{{real_type[0]}};
    cf_c[15] =  3.82683432e-01{{real_type[0]}};

    c_off = 4*wn;
    j = r_off + 8*(wn&1); 

    t_r[0]  = {{a1}}[j]    + cf_r[c_off]  * {{a1}}[j+1]  - cf_c[c_off]  * {{a2}}[j+1];
    t_r[1]  = {{a1}}[j+2]  + cf_r[c_off+1]* {{a1}}[j+3]  - cf_c[c_off+1]* {{a2}}[j+3];
    t_r[2]  = {{a1}}[j+4]  + cf_r[c_off+2]* {{a1}}[j+5]  - cf_c[c_off+2]* {{a2}}[j+5];
    t_r[3]  = {{a1}}[j+6]  + cf_r[c_off+3]* {{a1}}[j+7]  - cf_c[c_off+3]* {{a2}}[j+7];

    t_c[0]  = {{a2}}[j]    + cf_r[c_off]  * {{a2}}[j+1]  + cf_c[c_off]  * {{a1}}[j+1]; 
    t_c[1]  = {{a2}}[j+2]  + cf_r[c_off+1]* {{a2}}[j+3]  + cf_c[c_off+1]* {{a1}}[j+3];
    t_c[2]  = {{a2}}[j+4]  + cf_r[c_off+2]* {{a2}}[j+5]  + cf_c[c_off+2]* {{a1}}[j+5];
    t_c[3]  = {{a2}}[j+6]  + cf_r[c_off+3]* {{a2}}[j+7]  + cf_c[c_off+3]* {{a1}}[j+7];

    {{sync_fn}}

    j = r_off + c_off;
    {{a1}}[j]    = t_r[0];
    {{a1}}[j+1]  = t_r[1];
    {{a1}}[j+2]  = t_r[2];
    {{a1}}[j+3]  = t_r[3];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+1]  = t_c[1];
    {{a2}}[j+2]  = t_c[2];
    {{a2}}[j+3]  = t_c[3];

    {{sync_fn}}
""")


# FFT16, 128 workers, fast axis.
fft16_gs128_fa_tpl = jinja2.Template("""
    cf_r[0]  =  1.0{{real_type[0]}};
    cf_r[1]  =  9.23879533e-01{{real_type[0]}};
    cf_r[2]  =  7.07106781e-01{{real_type[0]}};
    cf_r[3]  =  3.82683432e-01{{real_type[0]}};
    cf_r[4]  =  0.0{{real_type[0]}};
    cf_r[5]  = -3.82683432e-01{{real_type[0]}};
    cf_r[6]  = -7.07106781e-01{{real_type[0]}};
    cf_r[7]  = -9.23879533e-01{{real_type[0]}};
    cf_r[8]  = -1.0{{real_type[0]}};
    cf_r[9]  = -9.23879533e-01{{real_type[0]}};
    cf_r[10] = -7.07106781e-01{{real_type[0]}};
    cf_r[11] = -3.82683432e-01{{real_type[0]}};
    cf_r[12] =  0.0{{real_type[0]}};
    cf_r[13] =  3.82683432e-01{{real_type[0]}};
    cf_r[14] =  7.07106781e-01{{real_type[0]}};
    cf_r[15] =  9.23879533e-01{{real_type[0]}};

    cf_c[0]  =  0.0{{real_type[0]}};
    cf_c[1]  = -3.82683432e-01{{real_type[0]}};
    cf_c[2]  = -7.07106781e-01{{real_type[0]}};
    cf_c[3]  = -9.23879533e-01{{real_type[0]}};
    cf_c[4]  = -1.0{{real_type[0]}};
    cf_c[5]  = -9.23879533e-01{{real_type[0]}};
    cf_c[6]  = -7.07106781e-01{{real_type[0]}};
    cf_c[7]  = -3.82683432e-01{{real_type[0]}};
    cf_c[8]  =  0.0{{real_type[0]}};
    cf_c[9]  =  3.82683432e-01{{real_type[0]}};
    cf_c[10] =  7.07106781e-01{{real_type[0]}};
    cf_c[11] =  9.23879533e-01{{real_type[0]}};
    cf_c[12] =  1.0{{real_type[0]}};
    cf_c[13] =  9.23879533e-01{{real_type[0]}};
    cf_c[14] =  7.07106781e-01{{real_type[0]}};
    cf_c[15] =  3.82683432e-01{{real_type[0]}};

    c_off = 2*wn;
    j = r_off + 4*(wn&3);

    t_r[0]  = {{a1}}[j]    + cf_r[c_off]  * {{a1}}[j+1]  - cf_c[c_off]  * {{a2}}[j+1];
    t_r[1]  = {{a1}}[j+2]  + cf_r[c_off+1]* {{a1}}[j+3]  - cf_c[c_off+1]* {{a2}}[j+3];

    t_c[0]  = {{a2}}[j]    + cf_r[c_off]  * {{a2}}[j+1]  + cf_c[c_off]  * {{a1}}[j+1]; 
    t_c[1]  = {{a2}}[j+2]  + cf_r[c_off+1]* {{a2}}[j+3]  + cf_c[c_off+1]* {{a1}}[j+3];

    {{sync_fn}}

    j = r_off + c_off;
    {{a1}}[j]    = t_r[0];
    {{a1}}[j+1]  = t_r[1];

    {{a2}}[j]    = t_c[0];
    {{a2}}[j+1]  = t_c[1];

    {{sync_fn}}
""")


# FFT16, 256 workers, fast axis.
fft16_gs256_fa_tpl = jinja2.Template("""
    cf_r[0]  =  1.0{{real_type[0]}};
    cf_r[1]  =  9.23879533e-01{{real_type[0]}};
    cf_r[2]  =  7.07106781e-01{{real_type[0]}};
    cf_r[3]  =  3.82683432e-01{{real_type[0]}};
    cf_r[4]  =  0.0{{real_type[0]}};
    cf_r[5]  = -3.82683432e-01{{real_type[0]}};
    cf_r[6]  = -7.07106781e-01{{real_type[0]}};
    cf_r[7]  = -9.23879533e-01{{real_type[0]}};
    cf_r[8]  = -1.0{{real_type[0]}};
    cf_r[9]  = -9.23879533e-01{{real_type[0]}};
    cf_r[10] = -7.07106781e-01{{real_type[0]}};
    cf_r[11] = -3.82683432e-01{{real_type[0]}};
    cf_r[12] =  0.0{{real_type[0]}};
    cf_r[13] =  3.82683432e-01{{real_type[0]}};
    cf_r[14] =  7.07106781e-01{{real_type[0]}};
    cf_r[15] =  9.23879533e-01{{real_type[0]}};

    cf_c[0]  =  0.0{{real_type[0]}};
    cf_c[1]  = -3.82683432e-01{{real_type[0]}};
    cf_c[2]  = -7.07106781e-01{{real_type[0]}};
    cf_c[3]  = -9.23879533e-01{{real_type[0]}};
    cf_c[4]  = -1.0{{real_type[0]}};
    cf_c[5]  = -9.23879533e-01{{real_type[0]}};
    cf_c[6]  = -7.07106781e-01{{real_type[0]}};
    cf_c[7]  = -3.82683432e-01{{real_type[0]}};
    cf_c[8]  =  0.0{{real_type[0]}};
    cf_c[9]  =  3.82683432e-01{{real_type[0]}};
    cf_c[10] =  7.07106781e-01{{real_type[0]}};
    cf_c[11] =  9.23879533e-01{{real_type[0]}};
    cf_c[12] =  1.0{{real_type[0]}};
    cf_c[13] =  9.23879533e-01{{real_type[0]}};
    cf_c[14] =  7.07106781e-01{{real_type[0]}};
    cf_c[15] =  3.82683432e-01{{real_type[0]}};

    c_off = wn;
    j = r_off + 2*(wn&7);

    t_r[0]  = {{a1}}[j]    + cf_r[c_off]  * {{a1}}[j+1]  - cf_c[c_off]  * {{a2}}[j+1];
    t_c[0]  = {{a2}}[j]    + cf_r[c_off]  * {{a2}}[j+1]  + cf_c[c_off]  * {{a1}}[j+1]; 

    {{sync_fn}}

    j = r_off + c_off;
    {{a1}}[j]    = t_r[0];
    {{a2}}[j]    = t_c[0];

    {{sync_fn}}
""")
