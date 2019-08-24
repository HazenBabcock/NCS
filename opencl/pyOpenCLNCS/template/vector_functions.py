#!/usr/bin/env python
#
# Templates for the vector functions.
#
# group_size - The number of workers in a group.
# item_size - The number of elements each worker is responsible for (this is a power of 2).
#
# Hazen 08/19
#

import jinja2


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

  if(lid == 0){
     for(i=1; i<{{group_size}}; i++){
        {{a1}}[0] += {{a1}}[i];
     }
  } 

  {{sync_fn}}

{% if not_inline %}
}
{% endif %}
""")

def vecdot(a1, a2, a3, args):
    tmp = args.copy()
    tmp["a1"] = a1
    tmp["a2"] = a2
    tmp["a3"] = a3
    return vecdot_tpl.render(tmp)


# Compare two vectors for equality.
vecisEqual_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecisEqual({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, int lid)
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

  if(lid == 0){
     for(i=1; i<{{group_size}}; i++){
        {{a1}}[0] += {{a1}}[i];
     }
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
    tmp["a3"] = a2
    tmp["a4"] = a2
    return veccopy_tpl.render(tmp)


# Float multiply and add in place,  a1 = a2 * a3 + a1.
vecfmaInplace_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecfma({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{real_type}} {{a3}}, int lid)
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
    tmp["a3"] = a2
    return vecfmaInplace_tpl.render(tmp)


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
