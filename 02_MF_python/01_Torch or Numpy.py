import torch
import numpy as np


np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

tensortoarray = torch_data.numpy()
print(
	np_data,
	tensortoarray
	)
# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
