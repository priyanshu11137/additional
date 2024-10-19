import torch

# Check if CUDA is available and get the GPU device number
if torch.cuda.is_available():
    gpu_device_count = torch.cuda.device_count()
    for i in range(gpu_device_count):
        print(f"GPU Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")
