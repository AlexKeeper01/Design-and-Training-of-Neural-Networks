import numpy as np


class HebbianWeather:
    def __init__(self, pattern_length, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.weights = np.zeros(pattern_length)
        self.bias = 0
        self.threshold = 0

    def train(self, patterns, labels):
        bipolar_labels = [2 * label - 1 for label in labels]

        for pattern, bipolar_label in zip(patterns, bipolar_labels):
            self.weights += bipolar_label * pattern * self.learning_rate
            self.bias += bipolar_label * self.learning_rate

        scores = []
        for pattern in patterns:
            score = np.dot(self.weights, pattern) + self.bias
            scores.append(score)

        good_scores = [s for s, l in zip(scores, labels) if l == 1]
        bad_scores = [s for s, l in zip(scores, labels) if l == 0]

        self.threshold = (min(good_scores) + max(bad_scores)) / 2

        print("Результаты обучения:")
        print(f"Веса: {self.weights}")
        print(f"Смещение: {self.bias}")
        print(f"Порог: {self.threshold:.2f}")
        print(f"Очки: {[float(round(s, 2)) for s in scores]}")

    def predict(self, pattern):
        score = np.dot(self.weights, pattern) + self.bias
        return 1 if score > self.threshold else 0


weather_patterns = [
    np.array([1, 0, 0, 0]),
    np.array([1, 1, 0, 0]),
    np.array([0, 1, 1, 1]),
    np.array([0, 0, 1, 1]),
    np.array([0, 0, 0, 0]),
]
labels = [1, 1, 0, 0, 1]

m = HebbianWeather(4)
m.train(weather_patterns, labels)

print("\nРезультаты тестирования:")
print("Формат: [солнце, облака, ветер, осадки] (1=есть, 0=нет)")
test_patterns = [
    np.array([1, 0, 0, 0]),
    np.array([1, 1, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 1, 1, 0]),
    np.array([1, 0, 1, 0]),
    np.array([0, 0, 1, 1]),
    np.array([0, 1, 1, 1]),
    np.array([0, 0, 0, 1]),
]

for pattern in test_patterns:
    score = np.dot(m.weights, pattern) + m.bias
    result = m.predict(pattern)
    print(f"{pattern} -> {result} (Очки: {score:.2f}, Порог: {m.threshold:.2f})")
