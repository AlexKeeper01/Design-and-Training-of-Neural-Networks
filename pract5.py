import numpy as np

# Входные данные
patterns = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
])
labels = np.array([1, 1, 0, 0, 1])

# Параметры SOM
grid_size = (3, 3)
input_dim = patterns.shape[1]
learning_rate = 0.5
radius = max(grid_size) / 2
epochs = 100

# Инициализация весов случайно
weights = np.random.rand(grid_size[0], grid_size[1], input_dim)


def euclidean(a, b):
    return np.linalg.norm(a - b)


for epoch in range(epochs):
    for pattern in patterns:
        # Находим ближайший нейрон (BMU)
        distances = np.linalg.norm(weights - pattern, axis=2)
        bmu_idx = np.unravel_index(distances.argmin(), distances.shape)

        # Обновляем веса BMU и соседей
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Расстояние до BMU на сетке
                d = np.sqrt((i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2)
                if d <= radius:
                    influence = np.exp(-d ** 2 / (2 * (radius ** 2)))
                    weights[i, j] += learning_rate * influence * (pattern - weights[i, j])
    # Постепенно уменьшаем learning rate и radius
    learning_rate *= 0.9
    radius *= 0.95

# После обучения можно "маркировать" нейроны по меткам
neuron_labels = np.zeros(grid_size)
for pattern, label in zip(patterns, labels):
    distances = np.linalg.norm(weights - pattern, axis=2)
    bmu_idx = np.unravel_index(distances.argmin(), distances.shape)
    neuron_labels[bmu_idx] = label

# Тестирование
test_patterns = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 1],
])

print("\nРезультаты тестирования:")
print("Формат: [солнце, облака, ветер, осадки] (1=есть, 0=нет)\n")

for pattern in test_patterns:
    distances = np.linalg.norm(weights - pattern, axis=2)
    bmu_idx = np.unravel_index(distances.argmin(), distances.shape)
    result = neuron_labels[bmu_idx]
    print(f"{pattern} -> {int(result)}")
