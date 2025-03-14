import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# 予測
prediction = model(data)

# lossを計算, .backward()でbackpropageteをする
loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 勾配降下を実施
optim.step()

# 以下のセクションは自動微分について説明
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

# QをQで微分すると、単位行列になる。ベクトルの方向を指定する必要がある
# 特定の勾配を求めることは、どの情報が重要かを選択する手段。らしい
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# print(9*a**2 == a.grad)
# print(-2*b == b.grad)

# torch.autogradはヤコビ行列Jを直接計算するのではなくて、任意のベクトルvとの積を計算するようになっている.J^T ・v
# ヤコビ行列はO(n*m) ベクトルーヤコビはO(n)

# autogradはDAGを根(出力)から葉(入力)へ遡ることによって、連鎖律を用いて自動的に勾配を計算している

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
# print(f"Does `a` require gradients?: {a.requires_grad}")
# b = x + z
# print(f"Does `b` require gradients?: {b.requires_grad}")

from torch import nn, optim

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
	param.requires_grad = False

# 最終層をfine-turingしてみる
model.fc = nn.Linear(512, 10)

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.2)
