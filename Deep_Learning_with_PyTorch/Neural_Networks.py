# 1.学習可能なパラメータを持つNNを定義する
# 2.入力でーたセットを反復処理する
# 3.入力でーたをネットワークに通す
# 4.損失を計算する
# 5.勾配をネットワークのパラメータに逆伝搬させる
# 6.ネットワークの重みを更新する

# 1.define network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, input):
		c1 = F.relu(self.conv1(input))
		s2 = F.max_pool2d(c1, (2, 2))
		c3 = F.relu(self.conv2(s2))
		s4 = F.max_pool2d(c3, 2)
		s4 = torch.flatten(s4, 1)
		f5 = F.relu(self.fc1(s4))
		f6 = F.relu(self.fc2(f5))
		output = self.fc3(f6)
		return output

net = Net()
# print(net)

# forward関数を作成すると自動的にbackward関数が定義される

# params = list(net.parameters())
# print(len(params))
# print(params[0].size()) # conv.1のweight torch.Size([6, 1, 5, 5])

# 以下のようになっている
# params = [
#     torch.Size([6, 1, 5, 5]),  # Conv1's weight
#     torch.Size([6]),            # Conv1's bias
#     torch.Size([16, 6, 5, 5]),  # Conv2's weight
#     torch.Size([16]),           # Conv2's bias
#     torch.Size([120, 400]),     # FC1's weight
#     torch.Size([120]),          # FC1's bias
#     torch.Size([84, 120]),      # FC2's weight
#     torch.Size([84]),           # FC2's bias
#     torch.Size([10, 84]),       # FC3's weight
#     torch.Size([10])            # FC3's bias
# ]

input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)

# 勾配はデフォルトで蓄積されるようになっているので前回の勾配をクリア
net.zero_grad()
out.backward(torch.randn(1, 10))

# ここまでで、NNの定義とbackwardの実行まで学習した
# 以下は、損失の計算方法と重みの更新について学習する

output = net(input)
target = torch.randn(10)
target = target.view(1, -1) # -1は残りの次元を自動的に決めてくれる
criterion = nn.MSELoss()

loss = criterion(output, target)
# print(loss)

# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> flatten -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# となっているので、loss.backwardを実行すると、このグラフに従って勾配が蓄積される
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()
# print("conf1.bias.grad before backward")
# print(net.conv1.bias.grad)

loss.backward()

# print("conv1.bias.grad after backward")
# print(net.conv1.bias.grad)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
