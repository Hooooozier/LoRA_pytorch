import torch
import numpy as np

torch.manual_seed(0)

d, k = 5, 5
W_rank = 2

W = torch.randn(d, W_rank) @ torch.randn(W_rank, k)

print(W)

W_rank = torch.linalg.matrix_rank(W)
print(f"ranking of W: {W_rank}")


U, S, V = torch.svd(W)
U_r = U[:, :W_rank]
S_r = torch.diag(S[:W_rank])
V_r = V[:, :W_rank].t()

# print(V_r)

# _, _, V = torch.linalg.svd(W)
# print(V[:W_rank, :])

B = U_r @ S_r
A = V_r


print(f"shape of A:{A.shape}")
print(f"shape of B:{B.shape}")


bias = torch.randn(d)
x = torch.randn(d)


y = W@x + bias
y_prime = (B @ A) @ x + bias
print(y)
print(y_prime)
