import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = np.array([1, 2, 3], dtype=np.float32)
print(x)
print(x.itemsize)
x = np.sin(x)
print(x)
x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(x.shape)
