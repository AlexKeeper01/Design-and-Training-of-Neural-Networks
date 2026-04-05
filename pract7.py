import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


temps = np.array([
    2.5, 3.1, 3.8, 4.2, 4.0, 3.6, 3.2,
    2.9, 3.3, 4.1, 5.0, 5.8, 6.3, 6.0,
    5.5, 5.2, 5.7, 6.4, 7.5, 8.3, 9.0,
    8.7, 8.1, 7.6, 7.0, 6.5, 5.9, 5.2,
    4.6, 4.0, 4.3, 5.0, 6.2, 7.5, 9.0,
    10.5, 11.7, 12.4, 12.0, 11.3, 10.6, 9.8,
    9.1, 8.3, 7.6, 7.9, 8.5, 9.2,
    10.0, 10.8, 11.5, 12.3, 13.0, 13.8, 13.2,
    12.5, 11.8, 11.0, 10.2, 9.5, 8.7, 8.0,
    7.2, 6.5, 5.8, 5.0, 4.3, 3.8, 4.2,
    5.0, 6.0, 7.2, 8.5, 9.8, 10.7, 11.5,
    12.2, 13.0, 13.8, 14.5, 15.2, 14.6, 13.9,
    13.0, 12.1, 11.2, 10.3, 9.5, 8.8, 8.0
])

mean = temps.mean()
std = temps.std()
temps_norm = (temps - mean) / std

seq_length = 14


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


X, y = create_sequences(temps_norm, seq_length)

X = torch.FloatTensor(X).unsqueeze(-1)
y = torch.FloatTensor(y)


# =========================
# 4. Модель RNN
# =========================
class TemperatureRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


model = TemperatureRNN()

# =========================
# 5. Обучение
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()

    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================
# 6. Прогноз
# =========================
model.eval()

last_seq = torch.FloatTensor(temps_norm[-seq_length:]).unsqueeze(0).unsqueeze(-1)

predictions = []
future_steps = 30

for _ in range(future_steps):
    with torch.no_grad():
        pred = model(last_seq)

    predictions.append(pred.item())

    last_seq = torch.cat([
        last_seq[:, 1:, :],
        pred.view(1, 1, 1)
    ], dim=1)

# Денормализация
predictions = np.array(predictions) * std + mean

# =========================
# 7. График
# =========================
plt.figure(figsize=(10, 5))

plt.plot(range(len(temps)), temps, label="История")
plt.plot(range(len(temps), len(temps) + future_steps), predictions, label="Прогноз")

plt.legend()
plt.grid()
plt.show()
