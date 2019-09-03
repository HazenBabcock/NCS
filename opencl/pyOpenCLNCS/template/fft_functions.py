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
    int i,j;

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
    int i,j;

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
    int i,j;

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
    int i,j;

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
    int i,j;

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

    {{a1}}[wn]   = t_r[0];
    {{a1}}[wn+4] = t_r[1];

    {{a2}}[wn]   = t_c[0];
    {{a2}}[wn+4] = t_c[1];
   
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
    t_r[0] = {{a1}}[j] + cf_r[c_off]*{{a1}}[j+4] - cf_c[c_off]*{{a2}}[j+4];
    t_c[0] = {{a2}}[j] + cf_c[c_off]*{{a1}}[j+4] + cf_r[c_off]*{{a2}}[j+4];

    {{sync_fn}}

    {{a1}}[lid] = t_r[0];
    {{a2}}[lid] = t_c[0];
   
    {{sync_fn}}
""")













fft4_tpl = jinja2.Template("""
    {{header}}

    t_r[0] = {{a1}}[i+0] + {{a1}}[i+4]; 
    t_r[1] = {{a1}}[i+8];
    t_r[2] = {{a1}}[i+0] - {{a1}}[i+4]; 
    t_r[3] = {{a1}}[i+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = -{{a1}}[i+12];
    t_c[2] = 0.0{{real_type[0]}};
    t_c[3] = {{a1}}[i+12];

    {{a1}}[i]    = t_r[0];
    {{a1}}[i+4]  = t_r[1];
    {{a1}}[i+8]  = t_r[2];
    {{a1}}[i+12] = t_r[3];

    {{a2}}[i]    = t_c[0];
    {{a2}}[i+4]  = t_c[1];
    {{a2}}[i+8]  = t_c[2];
    {{a2}}[i+12] = t_c[3];

""")


fft4_mp2_tpl = jinja2.Template("""
    {{real_type}} t_r[2];
    {{real_type}} t_c[2];

    {{real_type}} r4[2] = { 1.0{{real_type[0]}}, -1.0{{real_type[0]}}};
    {{real_type}} c4[2] = {-1.0{{real_type[0]}},  1.0{{real_type[0]}}};
 
    i = wn;
    c_off = wn&1;
    t_r[0] = {{a1}}[i]   + r4[c_off]*{{a1}}[i+4];
    t_r[1] = {{a1}}[i+8];

    t_c[0] = 0.0{{real_type[0]}};
    t_c[1] = c4[c_off]*{{a1}}[i+12];

    {{sync_fn}}

    i = wn+8*coff;
    {{a1}}[i]   = t_r[0];
    {{a1}}[i+4] = t_r[1];

    {{a2}}[i]   = t_c[0];
    {{a2}}[i+4] = t_c[1];

""")


fft4_mp4_tpl = jinja2.Template("""
    {{real_type}} t_r;
    {{real_type}} t_c;

    {{real_type}} r4[2] = { 1.0{{real_type[0]}},  0.0{{real_type[0]}}, -1.0{{real_type[0]}}, 0.0{{real_type[0]}}};
    {{real_type}} c4[2] = { 0.0{{real_type[0]}}, -1.0{{real_type[0]}},  0.0{{real_type[0]}}, 1.0{{real_type[0]}}};
 
    i = wn>>1;
    c_off = wn&3;
    t_r = {{a1}}[i] + r4[c_off]*{{a1}}[i+4];
    t_c =             c4[c_off]*{{a1}}[i+12];

    {{sync_fn}}

    {{a1}}[wn] = t_r;
    {{a2}}[wn] = t_c;

""")


# FFT4 for group size of 16, 32 and 64.
fft4_tpl = jinja2.Template("""
{{device}}void fft4({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int o)
{
    {{real_type}} t_r[4];
    {{real_type}} t_c[4];

    t_r[0] = {{a1}}[o] + {{a1}}[o+8];

    {{a2}}[o+0]  = {{a1}}[o+0] + {{a1}}[o+4] + {{a1}}[o+8] + {{a1}}[o+12];
    {{a2}}[o+4]  = {{a1}}[o+0] - {{a1}}[o+8];
    {{a2}}[o+8]  = {{a1}}[o+0] - {{a1}}[o+4] + {{a1}}[o+8] - {{a1}}[o+12];
    {{a2}}[o+12] = {{a1}}[o+0] - {{a1}}[o+8];

    {{a3}}[o+0]  = 0.0{{real_type[0]}};
    {{a3}}[o+4]  = -{{a1}}[o+4] + {{a1}}[o+12];
    {{a3}}[o+8]  = 0.0{{real_type[0]}};
    {{a3}}[o+12] = {{a1}}[o+4] - {{a1}}[o+12];
}
""")


# FFT4 for group size of 128.
fft4_128_tpl = jinja2.Template("""
{{device}}void fft4({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, {{real_type}} s1, int o1, int o2)
{
    {{a2}}[o1+0]  = {{a1}}[o2+0] + s1*{{a1}}[o2+4] + {{a1}}[o2+8] + s1*{{a1}}[o2+12];
    {{a2}}[o1+4]  = {{a1}}[o2+0] - {{a1}}[o2+8];

    {{a3}}[o1+0]  = 0.0{{real_type[0]}};
    {{a3}}[o1+4]  = s1*({{a1}}[o2+4] + {{a1}}[o2+12]);
}
""")



# FFT4 for group size of 256.
fft4_256_tpl = jinja2.Template("""
{{device}}void fft4({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, {{real_type}} s1, {{real_type}} s2, int o1, int o2)
{
    {{a2}}[o1] = {{a1}}[o2+0] + s1*{{a1}}[o2+4] + s2*{{a1}}[o2+8] + s1*{{a1}}[o2+12];

    {{a3}}[o1] = s1*s2*({{a1}}[o2+4] + {{a1}}[o2+12]);
}
""")



# FFT8 for group size of 16, 32.
fft8_tpl = jinja2.Template("""
{{device}}void fft8({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int o)
{
    float t_r[8];
    float t_c[8];

    t_r[0] = {{a1}}[o]    + {{a1}}[o+2];
    t_r[1] = {{a1}}[o+4]  + {{a1}}[o+6] *7.07106781e-01{{real_type[0]}} + {{a2}}[o+6] *7.07106781e-01{{real_type[0]}};
    t_r[2] = {{a1}}[o+8]                                                + {{a2}}[o+10];
    t_r[3] = {{a1}}[o+12] - {{a1}}[o+14]*7.07106781e-01{{real_type[0]}} + {{a2}}[o+14]*7.07106781e-01{{real_type[0]}};

    t_r[4] = {{a1}}[o]    - {{a1}}[o+2];
    t_r[5] = {{a1}}[o+4]  - {{a1}}[o+6] *7.07106781e-01{{real_type[0]}} - {{a2}}[o+6] *7.07106781e-01{{real_type[0]}};
    t_r[6] = {{a1}}[o+8]                                                - {{a2}}[o+10];
    t_r[7] = {{a1}}[o+12] + {{a1}}[o+14]*7.07106781e-01{{real_type[0]}} - {{a2}}[o+14]*7.07106781e-01{{real_type[0]}};

    t_c[0] = {{a2}}[o]    + {{a2}}[o+2];
    t_c[1] = {{a2}}[o+4]  + {{a2}}[o+6] *7.07106781e-01{{real_type[0]}} - {{a1}}[o+6] *7.07106781e-01{{real_type[0]}};
    t_c[2] = {{a2}}[o+8]                                                - {{a1}}[o+10];
    t_c[3] = {{a2}}[o+12] - {{a2}}[o+14]*7.07106781e-01{{real_type[0]}} - {{a1}}[o+14]*7.07106781e-01{{real_type[0]}};

    t_c[4] = {{a2}}[o]    - {{a2}}[o+2];
    t_c[5] = {{a2}}[o+4]  - {{a2}}[o+6] *7.07106781e-01{{real_type[0]}} + {{a1}}[o+6] *7.07106781e-01{{real_type[0]}};
    t_c[6] = {{a2}}[o+8]                                                + {{a1}}[o+10];
    t_c[7] = {{a2}}[o+12] + {{a2}}[o+14]*7.07106781e-01{{real_type[0]}} + {{a1}}[o+14]*7.07106781e-01{{real_type[0]}};

    {{sync_fn}}

    {{a1}}[o+0]  = t_r[0];
    {{a1}}[o+2]  = t_r[1];
    {{a1}}[o+4]  = t_r[2];
    {{a1}}[o+6]  = t_r[3];
    {{a1}}[o+8]  = t_r[4];
    {{a1}}[o+10] = t_r[5];
    {{a1}}[o+12] = t_r[6];
    {{a1}}[o+14] = t_r[7];

    {{a2}}[o+0]  = t_c[0];
    {{a2}}[o+2]  = t_c[1];
    {{a2}}[o+4]  = t_c[2];
    {{a2}}[o+6]  = t_c[3];
    {{a2}}[o+8]  = t_c[4];
    {{a2}}[o+10] = t_c[5];
    {{a2}}[o+12] = t_c[6];
    {{a2}}[o+14] = t_c[7];
}
""")


# FFT8 for group size of 64.
fft8_64_tpl = jinja2.Template("""
{{device}}void fft8({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{real_type}} s1, int o1, int o2)
{
    float t_r[4];
    float t_c[4];

    t_r[0] = {{a1}}[o1]    + s1*{{a1}}[o1+2];
    t_r[1] = {{a1}}[o1+4]  + s1*{{a1}}[o1+6] *7.07106781e-01{{real_type[0]}} + s1*{{a2}}[o1+6] *7.07106781e-01{{real_type[0]}};
    t_r[2] = {{a1}}[o1+8]                                                    + s1*{{a2}}[o1+10];
    t_r[3] = {{a1}}[o1+12] - s1*{{a1}}[o1+14]*7.07106781e-01{{real_type[0]}} + s1*{{a2}}[o1+14]*7.07106781e-01{{real_type[0]}};

    t_c[0] = {{a2}}[o1]    + s1*{{a2}}[o1+2];
    t_c[1] = {{a2}}[o1+4]  + s1*{{a2}}[o1+6] *7.07106781e-01{{real_type[0]}} - s1*{{a1}}[o1+6] *7.07106781e-01{{real_type[0]}};
    t_c[2] = {{a2}}[o1+8]                                                    - s1*{{a1}}[o1+10];
    t_c[3] = {{a2}}[o1+12] - s1*{{a2}}[o1+14]*7.07106781e-01{{real_type[0]}} - s1*{{a1}}[o1+14]*7.07106781e-01{{real_type[0]}};

    {{sync_fn}}

    {{a1}}[o2+0]  = t_r[0];
    {{a1}}[o2+2]  = t_r[1];
    {{a1}}[o2+4]  = t_r[2];
    {{a1}}[o2+6]  = t_r[3];

    {{a2}}[o2+0]  = t_c[0];
    {{a2}}[o2+2]  = t_c[1];
    {{a2}}[o2+4]  = t_c[2];
    {{a2}}[o2+6]  = t_c[3];
}
""")


# FFT8 for group size of 128.
fft8_128_tpl = jinja2.Template("""
{{device}}void fft8({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{real_type}} *r1, {{real_type}} *c1, int o1, int o2)
{
    float t_r[2];
    float t_c[2];

    t_r[0] = {{a1}}[o1]    + s1*{{a1}}[o1+2];
    t_r[1] = {{a1}}[o1+4]  + s1*{{a1}}[o1+6] *7.07106781e-01{{real_type[0]}} + s1*{{a2}}[o1+6] *7.07106781e-01{{real_type[0]}};
    t_r[2] = {{a1}}[o1+8]                                                    + s1*{{a2}}[o1+10];
    t_r[3] = {{a1}}[o1+12] - s1*{{a1}}[o1+14]*7.07106781e-01{{real_type[0]}} + s1*{{a2}}[o1+14]*7.07106781e-01{{real_type[0]}};

    t_c[0] = {{a2}}[o1]    + s1*{{a2}}[o1+2];
    t_c[1] = {{a2}}[o1+4]  + s1*{{a2}}[o1+6] *7.07106781e-01{{real_type[0]}} - s1*{{a1}}[o1+6] *7.07106781e-01{{real_type[0]}};
    t_c[2] = {{a2}}[o1+8]                                                    - s1*{{a1}}[o1+10];
    t_c[3] = {{a2}}[o1+12] - s1*{{a2}}[o1+14]*7.07106781e-01{{real_type[0]}} - s1*{{a1}}[o1+14]*7.07106781e-01{{real_type[0]}};

    {{sync_fn}}

    {{a1}}[o2+0]  = t_r[0];
    {{a1}}[o2+2]  = t_r[1];
    {{a1}}[o2+4]  = t_r[2];
    {{a1}}[o2+6]  = t_r[3];

    {{a2}}[o2+0]  = t_c[0];
    {{a2}}[o2+2]  = t_c[1];
    {{a2}}[o2+4]  = t_c[2];
    {{a2}}[o2+6]  = t_c[3];
}
""")


#
# Function templates.
#

# Vector copy.
veccopy_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void veccopy({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = {{a2}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")


