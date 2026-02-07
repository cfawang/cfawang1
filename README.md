import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- 数论相关特征函数 ----
def crt_encode(n, primes):
    """
    用中国剩余定理(CRT)编码整数 n：
    返回 n mod p1, n mod p2, ..., n mod pk
    primes: 质数列表
    """
    return [n % p for p in primes]

# ---- 数据构造 ----
class NumberDataset(Dataset):
    def __init__(self, nums, primes, max_len):
        self.data = nums
        self.primes = primes
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        n = self.data[idx]
        x = crt_encode(n, self.primes)
        # pad/crt vector to uniform length
        x = x + [0]*(self.max_len - len(x))
        # 标签：1 表示是素数，否则 0
        y = 1 if all(n % p != 0 for p in range(2, int(n**0.5)+1)) else 0
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float)

# ---- Transformer 模型 ----
class NumTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)            # embed features
        x = self.transformer(x)      # transformer 编码
        # 池化取平均
        x = x.mean(dim=1)
        return torch.sigmoid(self.fc(x))

# ---- 训练与测试 ----
def train():
    primes = [2,3,5,7,11,13,17,19,23,29]  # 一组用来编码特征的质数
    nums = list(range(100, 2000))        # 样本整数范围
    dataset = NumberDataset(nums, primes, len(primes))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NumTransformer(vocab_size=2000, d_model=32, nhead=4, num_layers=2)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    return model

if __name__ == "__main__":
    model = train()
    # 简单 demo: 预测 4321 是否可能是素数
    inputs = torch.tensor([crt_encode(4321, [2,3,5,7,11,13,17,19,23,29])], dtype=torch.long)
    with torch.no_grad():
        print("Prob prime ≈", model(inputs).item())
