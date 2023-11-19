import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
# print(my_tensor)

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device="cuda", requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

# other common initialization methods
x = torch.empty(size= (3, 3))
x = torch.zeros(3, 3)
x = torch.ones((3, 3))
x = torch.eye(5, 5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.diag(torch.ones(3))
# print(x)

# how to initialize and convert tensors to other types (int, float, double)
# tensor = torch.arange(4)
# print(tensor.bool())
# print(tensor.short())
# print(tensor.long())
# print(tensor.half())
# print(tensor.float())
# print(tensor.double())

# array to tensor conversion and vice_versa
# import numpy as np
# np_array = np.zeros((5, 5))
# print(np_array)
# tensor = torch.from_numpy(np_array)
# print(tensor)
# np_array_back = tensor.numpy()
# print(np_array)

# tensor math and comparison operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z = x + y
print(z)

z = x - y
print(z)

z = x ** 2
print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)


# matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise mult.
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm)