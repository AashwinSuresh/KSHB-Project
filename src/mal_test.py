import paddle
print("Compiled with CUDA:", paddle.is_compiled_with_cuda())
print("GPU device count:", paddle.device.cuda.device_count())