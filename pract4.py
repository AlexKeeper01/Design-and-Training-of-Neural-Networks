import torch
import torch.nn as nn
import torch.optim as optim


class RBFNet(nn.Module):
    def __init__(self, in_features, num_centers):
        super().__init__()

        self.num_centers = num_centers

        # центры
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))

        # ширина гауссианы (можно сделать обучаемой)
        self.sigma = 1.0

        # выходной слой
        self.linear = nn.Linear(num_centers, 1)

    def rbf(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)                 # (batch, 1, features)
        centers = self.centers.unsqueeze(0)  # (1, num_centers, features)

        distances = torch.sum((x - centers) ** 2, dim=2)

        return torch.exp(-distances / (2 * self.sigma ** 2))

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return torch.sigmoid(x)


# ====== ДАННЫЕ ======
weather_patterns = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
], dtype=torch.float32)

labels = torch.tensor([[1], [1], [0], [0], [1]], dtype=torch.float32)


# ====== МОДЕЛЬ ======
model = RBFNet(in_features=4, num_centers=5)

# 👉 инициализация центров из данных (лучше, чем случайно)
with torch.no_grad():
    model.centers.copy_(weather_patterns)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 200


# ====== ОБУЧЕНИЕ (батчем, без ошибок размеров) ======
for epoch in range(epochs):
    y_pred = model(weather_patterns)   # (5, 1)
    loss = criterion(y_pred, labels)   # (5, 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ====== ПОРОГ ======
with torch.no_grad():
    scores = model(weather_patterns).squeeze().numpy()

good_scores = [s for s, l in zip(scores, labels) if l == 1]
bad_scores = [s for s, l in zip(scores, labels) if l == 0]

threshold = (min(good_scores) + max(bad_scores)) / 2

print("Порог:", round(float(threshold), 2))
print("Очки:", [round(float(s), 2) for s in scores])


# ====== ТЕСТ ======
print("\nРезультаты тестирования:")
print("Формат: [солнце, облака, ветер, осадки] (1=есть, 0=нет)\n")

test_patterns = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 1],
], dtype=torch.float32)

with torch.no_grad():
    test_scores = model(test_patterns).squeeze().numpy()

for pattern, score in zip(test_patterns, test_scores):
    result = 1 if score > threshold else 0
    print(f"{pattern.numpy()} -> {result} (Очки: {score:.2f}, Порог: {threshold:.2f})")
