import torch
x = torch.randn(40, 1, 50)
print('dimension', x.ndim)
print(len(x.view(len(x), 1, -1)))
print(len(x.view(len(x), 1, -1)[0]))
