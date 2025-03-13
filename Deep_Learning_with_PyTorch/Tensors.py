import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print(x_np)

x_ones = torch.ones_like(x_data)
# print(f"Ones Tenser: \n{x_ones}\n") # x_dataの形状を保持
x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n{x_rand}\n") # x_dataの形状を保持してrandでオーバーライド

shape = (2, 3,) # (2,)のようにすることでtupleになる。これに統一するために(2, 3,)としているのだと思う
rand_tensor = torch.rand(2,3) # 単にこれでも良い
ones_tenser = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(rand_tensor)
# print(ones_tenser)
# print(zeros_tensor)

tensor = torch.rand(3, 4)

# print(tensor.shape)
# print(tensor.dtype)
# print(tensor.device)

if torch.cuda.is_available():
	tensor = torch.to('cuda')
	print(f"Device tensor is stored on : {tensor.device}")
else:
	print(f"Device tensor is stored on : {tensor.device}")
