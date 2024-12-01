import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Создание синтетического временного ряда
np.random.seed(42)
t = np.arange(1, 37)  # Временные точки
data = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, len(t))

# Функция для вычисления ошибки
def holt_winters_error(params, data, m):
    alpha1, alpha2, alpha3 = params
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonality = np.zeros(m)
    fitted = np.zeros(n)

    level[0] = data[0]
    trend[0] = data[1] - data[0]
    seasonality[:m] = data[:m] / np.mean(data[:m])

    for t in range(1, n):
        season_idx = (t - m) % m
        level[t] = alpha1 * (data[t] / seasonality[season_idx]) + (1 - alpha1) * (level[t - 1] + trend[t - 1])
        trend[t] = alpha2 * (level[t] - level[t - 1]) + (1 - alpha2) * trend[t - 1]
        seasonality[season_idx] = alpha3 * (data[t] / level[t]) + (1 - alpha3) * seasonality[season_idx]
        fitted[t] = (level[t - 1] + trend[t - 1]) * seasonality[season_idx]

    return np.mean((data - fitted) ** 2)

# Оптимизация параметров
m = 12  # Длина сезона
initial_params = [0.1, 0.1, 0.1]
bounds = [(0, 1), (0, 1), (0, 1)]
result = minimize(holt_winters_error, initial_params, args=(data, m), bounds=bounds, method='L-BFGS-B')
alpha1, alpha2, alpha3 = result.x

# Построение модели
n = len(data)
level = np.zeros(n)
trend = np.zeros(n)
seasonality = np.zeros(m)
forecast = np.zeros(n + 4)

level[0] = data[0]
trend[0] = data[1] - data[0]
seasonality[:m] = data[:m] / np.mean(data[:m])

for t in range(1, n):
    season_idx = (t - m) % m
    level[t] = alpha1 * (data[t] / seasonality[season_idx]) + (1 - alpha1) * (level[t - 1] + trend[t - 1])
    trend[t] = alpha2 * (level[t] - level[t - 1]) + (1 - alpha2) * trend[t - 1]
    seasonality[season_idx] = alpha3 * (data[t] / level[t]) + (1 - alpha3) * seasonality[season_idx]
    forecast[t] = (level[t - 1] + trend[t - 1]) * seasonality[season_idx]

for t in range(n, n + 4):
    season_idx = (t - m) % m
    forecast[t] = (level[-1] + (t - n + 1) * trend[-1]) * seasonality[season_idx]

# Вывод параметров
print("Оптимальные параметры сглаживания:")
print(f"Alpha1 (уровень): {alpha1:.4f}")
print(f"Alpha2 (тренд): {alpha2:.4f}")
print(f"Alpha3 (сезонность): {alpha3:.4f}")

# Коэффициенты парной регрессии
a = trend[0]
b = level[0]
print("\nКоэффициенты парной регрессии:")
print(f"Коэффициент наклона (a): {a:.4f}")
print(f"Свободный член (b): {b:.4f}")

# Сезонные индексы
print("\nСезонные индексы:")
print(np.round(seasonality, 4))

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(data)), data, label="Исходные данные", color="blue")
plt.plot(np.arange(len(data)), forecast[:len(data)], label="Сглаженный ряд", color="orange")
plt.plot(
    np.arange(len(data) - 1, len(data) + 4),
    np.concatenate(([forecast[len(data) - 1]], forecast[-4:])),
    label="Прогноз",
    color="green",
    linestyle="--",
)
plt.legend()
plt.title("Модель Хольта-Уинтерса: сглаживание и прогноз")
plt.xlabel("Временные точки")
plt.ylabel("Значения ряда")
plt.grid()
plt.show()
