import torch

cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

if cuda_available:
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
else:
    print("PyTorch is running with CPU-only support. CUDA is not active.")

