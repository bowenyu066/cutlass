import sys
sys.path.insert(0, "/home/fangwen/miniforge3/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages")

# import argparse
# import torch
# import time
# from typing import Type

# import cuda.bindings.driver as cuda

# import cutlass
# import cutlass.cute as cute
# from cutlass.cute.runtime import from_dlpack
# import cutlass.torch as cutlass_torch

# @cute.jit
# def test(a_tensor: cute.Tensor):
#     thr_layout = cute.make_ordered_layout((2, 2), order=(1, 0))
#     val_layout = cute.make_ordered_layout((2, 2), order=(1, 0))
#     tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
#     gA = cute.zipped_divide(a_tensor, tiler_mn)
#     # idC = cute.make_identity_tensor((8, 16))
#     # cC = cute.zipped_divide(idC, (4, 4))
#     cute.print_tensor(gA)

# def main():
#     torch_dtype = cutlass_torch.dtype(cutlass.Float32)
#     a = torch.arange(0, 128, dtype=torch_dtype, device=torch.device("cuda")).reshape(8, 16)
#     a_tensor = from_dlpack(a).mark_layout_dynamic()
#     compiled_func = cute.compile(test, a_tensor)
#     compiled_func(a_tensor)

# @cute.jit
# def print_layout():
#     layout = cute.make_layout(shape=((4,4),(2,4)), stride=((16,1),(64,4)))

# if __name__ == "__main__":
#     compiled_func = cute.compile(print_layout)
#     compiled_func()

# import cutlass
# import cutlass.cute as cute

# @cute.jit
# def test():
#     a = cute.make_layout((88, 8))
#     for i in range(cute.size(a)):
#         # This hangs!
#         # cute.printf("coord: {}, natural coord: {}, idx: {}", i, a.get_hier_coord(i), a(i))

#         # This works:
#         print("coord: {}, natural coord: {}, idx: {}".format(i, a.get_hier_coord(i), a(i)))

# test()

import cutlass
import cutlass.cute as cute

@cute.jit  # preprocessor=True (默认)
def test_with_preprocessor():
    print("[COMPILE] With preprocessor")
    for i in range(3):
        cute.printf("[RUNTIME] Preprocessed line %d\n", i)

def main():
    print("=== With Preprocessor ===")
    test_with_preprocessor()

if __name__ == "__main__":
    main()