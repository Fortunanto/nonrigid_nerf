import torch
torch.cuda.is_available()
print(torch.cuda.memory_summary(0))