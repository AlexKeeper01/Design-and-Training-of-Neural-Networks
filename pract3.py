import torch
import torch.nn as nn
import torch.optim as optim


class WeatherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


weather_patterns = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
], dtype=torch.float32)

labels = torch.tensor([[1], [1], [0], [0], [1]], dtype=torch.float32)


model = WeatherNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)
epochs = 100

for epoch in range(epochs):
    for x, y in zip(weather_patterns, labels):
        y_pred = model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


with torch.no_grad():
    scores = model(weather_patterns).squeeze().numpy()

good_scores = [s for s, l in zip(scores, labels) if l == 1]
bad_scores = [s for s, l in zip(scores, labels) if l == 0]

threshold = (min(good_scores) + max(bad_scores)) / 2

print("Порог:", round(float(threshold), 2))
print("Очки:", [round(float(s), 2) for s in scores])


print("\nРезультаты тестирования:")

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
