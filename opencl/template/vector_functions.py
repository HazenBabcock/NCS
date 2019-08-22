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
{{device}}void veccopy({{local}}{{real_type}} *{{v1}}, {{local}}{{real_type}} *{{v2}}, int lid)
{
  int i = lid*{{item_size}};
  {%- endif -%}

  {% for i in indices %}
  {{v1}}[{{i}}] = {{v2}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def veccopy(a1, a2, args):
    args["a1"] = a1
    args["a2"] = a2
    return veccopy_tpl.render(args)


# Vector negative copy.
vecncopy_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecncopy({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, int lid)
{
  int i = lid*{{item_size}};
  {%- endif -%}

  {% for i in indices %}
  {{a1}}[{{i}}] = -{{a2}}[{{i}}];
  {%- endfor -%}

{% if not_inline %}
}
{% endif %}
""")

def vecncopy(a1, a2, args):
    args["a1"] = a1
    args["a2"] = a2
    return vecncopy_tpl.render(args)


# Vector dot product (stored in the first element of a1).
vecdot_tpl = jinja2.Template("""
{%- if not_inline -%}
{{device}}void vecdot({{local}}{{real_type}} *{{a1}}, {{local}}{{real_type}} *{{a2}}, {{local}}{{real_type}} *{{a3}}, int lid)
{
  int i = lid*{{scale}};
  {{real_type}} sum = ({{real_type}})0.0;
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
    args["a1"] = a1
    args["a2"] = a2
    args["a3"] = a3
    return vecdot_tpl.render(args)

