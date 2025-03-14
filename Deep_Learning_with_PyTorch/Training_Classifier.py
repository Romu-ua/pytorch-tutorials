# 1. CIFER10のデータセットを読み込んで正規化
# 2. CNNを定義
# 3. loss funcを定義
# 4. 訓練
# 5. テスト

import torch
import torchvision
import torchvision.transforms as transforms

# detaloaderを使うときにmainで保護してないと怒られる
def main():

	# torchvisionのデータセットは[0, 1]。[-1, 1]に正規化
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # transforms.Normalize(mean, std) [0, 1]->[-1, 1]
	])
	batch_size = 4

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# データセットを見てみる
	import matplotlib.pyplot as plt
	import numpy as np

	def imshow(img):
		img = img / 2 + 0.5
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
	dataiter = iter(trainloader)
	images, labels = next(dataiter)

	imshow(torchvision.utils.make_grid(images))
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

	import torch.nn as nn
	import torch.nn.functional as F


	class Net(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, 6, 5)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(16 * 5 * 5, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

		def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = torch.flatten(x, 1) # flatten all dimensions except batch
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x

	net = Net()

	import torch.optim as optim

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# データ全体を使用する回数(epoch)は２
	for epoch in range(2):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2000 == 1999:
				print(f"[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 2000:.3f}")
				running_loss = 0.0
	print("finished training")

	PATH = './cifar_net.pth'
	torch.save(net.state_dict(), PATH)

	dataiter = iter(testloader)
	images, labels = next(dataiter)

	# print images
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

if __name__ == "__main__":
	main()
