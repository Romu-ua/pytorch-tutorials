import torch 
import torch.nn as nn
import torch.optim as optim

# 語彙のサイズ10, 埋め込み次元5
embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)
optimizer = optim.SGD(embedding.parameters(), lr=0.1)


input_ids = torch.tensor([1, 5, 7])
target_vector = torch.rand(3, 5)

output_vectors = embedding(input_ids)
loss = nn.MSELoss()(output_vectors, target_vector)
loss.backward()
optimizer.step()
print(embedding.weight)

import sys
print(sys.executable)
