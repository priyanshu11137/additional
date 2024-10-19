import torch
print(torch.cuda.memory_summary())

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
