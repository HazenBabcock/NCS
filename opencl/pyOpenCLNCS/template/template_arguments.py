#!/usr/bin/env python
#
# Dictionary of arguments for generating functions from templates.
#
# Hazen 08/19
#


size_to_depth = {16 : 4, 32 : 5, 64 : 6, 128 : 7, 256 : 8}

def addOpenCL(args):
    """
    Add arguments for OpenCL kernel code.
    """
    args["local"] = "__local "
    args["sync_fn"] = "barrier(CLK_LOCAL_MEM_FENCE);"
    
    
def arguments(work_group_size, not_inline = True, real_type = "float"):
    """
    work_group_size is the work group or grid size.
    """

    # Check work_group_size.
    assert (work_group_size in size_to_depth), str(work_group_size) + " is not supported!"

    # This is the number of elements each thread is responsible for.
    item_size = int(256/work_group_size)
    
    # Indices to use for un-rolled loops.
    indices = []
    if (item_size>1):
        for i in range(item_size):
            if (i>0):
                indices.append("i+{0:d}".format(i))
            else:
                indices.append("i")
    else:
        indices.append("lid")

    args = {"depth" : size_to_depth[work_group_size],
            "indices" : indices,
            "item_size" : str(item_size),
            "not_inline" : not_inline,
            "real_type" : real_type}

    return args

            
if (__name__ == "__main__"):
    
    import pyOpenCLNCS.template.vector_functions as vf
    
    args = arguments(32)
    addOpenCL(args)

    print()
    print(vf.vecdot("v1", "v2", "v3", args))
    print()
