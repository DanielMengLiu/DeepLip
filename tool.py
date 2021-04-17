import torch

a = torch.rand(4,2)
print(a)

c = torch.arange(a.size(0))! 
a = a[c]
print(a)