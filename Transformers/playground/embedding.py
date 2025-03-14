import torch 
import torch.nn as nn
import torch.optim as optim

"""
使用方法
単語（またはサブワード）をトークン化
→ "Hello" → tokenizer("Hello") → [1045]
トークンIDを埋め込みベクトルに変換
→ [1045] → embedding(torch.tensor([1045])) → tensor([-0.8, 0.3, ...])
Transformer に入力し、系列関係を学習
逆伝播で埋め込み行列（embedding.weight）も更新

"""

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
