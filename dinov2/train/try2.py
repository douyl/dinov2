import torch
from xformers.ops import index_select_cat

# Define source tensors
source1 = torch.tensor([[1, 2], [3, 4], [5, 6]], device='cuda')
source2 = torch.tensor([[7, 8], [9, 10], [11, 12]], device='cuda')

# Define index tensors
index1 = torch.tensor([0, 2], device='cuda')
index2 = torch.tensor([1], device='cuda')

# Use index_select_cat to concatenate selected indices
output = index_select_cat([source1, source2], [index1, index2])
print(output)