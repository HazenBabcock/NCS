#!/usr/bin/env python
#
# Templates for the vector functions.
#
# group_size - The number of workers in a group (this must be one of 16,32,64,128 or 256).
# item_size - The number of elements each worker is responsible for (this is a power of 2).
#
# Hazen 08/19
#

import jinja2


#
# Helper templates.
#


# Vector sum (this is always inline).
sum_tpl = jinja2.Template("""
  if(lid < {{lid_max}}){
    {{a2}} = {{a1}}[2*lid] + {{a1}}[2*lid+1];
  }
  {{sync_fn}}
  {{a1}}[lid] = {{a2}};
  {{sync_fn}}

""")

def vecsum(a1, a2, args, is_float):
    tmp = args.copy()
    tmp["a1"] = a1;
    tmp["a2"] = a2;

    sum_str = "  " + a2 + " = 0"
    if is_float:
        sum_str += ".0" + args["real_type"][0]
    sum_str += ";"

    lid_max = []
    for i in range(args["depth"]):
        lid_max.append(2**i)
    lid_max.reverse()

    for elt in lid_max:
        tmp["lid_max"] = str(elt)
        sum_str += sum_tpl.render(tmp)

    return sum_str

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

def veccopy(a1, a2, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    return veccopy_tpl.render(tmp)


# Vector negative copy.
vecncopy_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecncopy({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = -{{a2}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecncopy(a1, a2, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    return vecncopy_tpl.render(tmp)


# Vector dot product (stored in the first element of a1).
vecdot_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecdot({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {% endif %}
  {{real_type}} sum = 0.0{{real_type[0]}};
  {%- endif -%}

  {% for i in indices %}
  sum += dot({{a2}}[{{i}}], {{a3}}[{{i}}]);
  {%- endfor %}

  {{a1}}[lid] = sum;

  {{sync_fn}}

  {{sum_fn}}

{% if not_inline %}
}
{% endif %}
""")

def vecdot(a1, a2, a3, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    tmp["a3"] = a3
    tmp["sum_fn"] = vecsum(a1, "sum", args, True)
    return vecdot_tpl.render(tmp)


# Compare two vectors for equality.
vecisEqual_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecisEqual({{local}}int *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {% endif %}
  int sum = 0;
  {%- endif -%}

  {% for i in indices %}
  sum += ({{a2}}[{{i}}] != {{a3}}[{{i}}]);
  {%- endfor %}

  {{a1}}[lid] = sum;

  {{sync_fn}}

  {{sum_fn}}

  if (lid==0){
    {{a1}}[0] = !{{a1}}[0];
  }

  {{sync_fn}}

{% if not_inline %}
}
{% endif %}
""")

def vecisEqual(a1, a2, a3, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    tmp["a3"] = a3
    tmp["sum_fn"] = vecsum(a1, "sum", args, False)
    return vecisEqual_tpl.render(tmp)


# Float multiply and add, a1 = a2 * a4 + a3 
vecfma_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecfma({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, {{real_type}} {{a4}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = fma({{a4}}, {{a2}}[{{i}}], {{a3}}[{{i}}]);
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecfma(a1, a2, a3, a4, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    tmp["a3"] = a3
    tmp["a4"] = a4
    return vecfma_tpl.render(tmp)


# Float multiply and add in place,  a1 = a2 * a3 + a1.
vecfmaInplace_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecfmaInplace({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{real_type}} {{a3}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = fma({{a3}}, {{a2}}[{{i}}], {{a1}}[{{i}}]);
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecfmaInplace(a1, a2, a3, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    tmp["a3"] = a3
    return vecfmaInplace_tpl.render(tmp)


# Calculate norm of a vector.
vecnorm_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecnorm({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int lid)
{
  {% endif %}
  vecdot({{a1}}, {{a2}}, {{a2}});

  {{sync_fn}}

  if (lid == 0){
    {{a1}}[0] = sqrt({{aq}}[0]);
  } 

  {{sync_fn}}

{% if not_inline %}
}
{% endif %}
""")

def vecnorm(a1, a2, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    return vecnorm_tpl.render(tmp)


# Scale vector in place.
vecscaleInplace_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecscaleInplace({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = {{a2}}[{{i}}] * {{a1}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecscaleInplace(a1, a2, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    return vecscaleInplace_tpl.render(tmp)


# Subtract two vectors.
vecsub_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecsub({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, int lid)
{
  {% if (item_size != "1") -%}
  int i = lid*{{item_size}};
  {%- endif -%}
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = {{a2}}[{{i}}] - {{a1}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecsub(a1, a2, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    return vecsub_tpl.render(tmp)
