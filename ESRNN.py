# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11LdNQbRBMkY4E7wOU8TJ-xaYr6q0t-nQ
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import os

# Константы
SEQ_LENGTH = 30
PRED_LENGTH = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MODEL_PATH = 'esrnn_crypto_returns.pth'

def load_crypto_data(ticker='BTC-USD', start='2020-01-01', end='2023-12-31'):
    print(f"Загрузка данных {ticker} с {start} по {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)

    # Лог-доходности и индикаторы
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['SMA_20_Returns'] = data['Log_Returns'].rolling(20).mean()
    data['EMA_10_Returns'] = data['Log_Returns'].ewm(span=10, adjust=False).mean()
    data['Volatility'] = data['Log_Returns'].rolling(20).std()
    data = data.dropna()

    features = data[['Log_Returns', 'SMA_20_Returns', 'EMA_10_Returns', 'Volatility']]
    return features.values, data

def create_sequences(data, dates, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH):
    X, y, X_dates = [], [], []
    for i in range(len(data)-seq_length-pred_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length, 0])  # Прогноз Log_Returns
        X_dates.append(dates[i+seq_length])
    return np.array(X), np.array(y), np.array(X_dates)

class ExponentialSmoothing(nn.Module):
    """Дифференцируемое экспоненциальное сглаживание с трендом и сезонностью"""
    def __init__(self, seasonality_period=7):
        super().__init__()
        self.seasonality_period = seasonality_period

        # Обучаемые параметры сглаживания
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Уровень
        self.beta = nn.Parameter(torch.tensor(0.1))   # Тренд
        self.gamma = nn.Parameter(torch.tensor(0.1))  # Сезонность

    def forward(self, x):
        """
        x: (batch, seq_length, 1) - временной ряд
        Возвращает: (batch, seq_length, 3) - уровень, тренд, сезонность
        """
        batch_size, seq_length, _ = x.shape
        device = x.device

        # Ограничиваем параметры в диапазоне [0, 1]
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.sigmoid(self.gamma)

        # Инициализация компонент
        level = torch.zeros(batch_size, seq_length, device=device)
        trend = torch.zeros(batch_size, seq_length, device=device)
        seasonal = torch.zeros(batch_size, seq_length, device=device)

        # Начальные значения
        level[:, 0] = x[:, 0, 0]
        if seq_length > 1:
            trend[:, 0] = x[:, 1, 0] - x[:, 0, 0]

        # Инициализация сезонности первым периодом
        for i in range(min(self.seasonality_period, seq_length)):
            seasonal[:, i] = x[:, i, 0] - level[:, 0]

        # Рекурсивное вычисление компонент
        for t in range(1, seq_length):
            if t >= self.seasonality_period:
                season_idx = t - self.seasonality_period
                prev_seasonal = seasonal[:, season_idx]
            else:
                prev_seasonal = seasonal[:, t]

            # Обновление уровня
            level[:, t] = alpha * (x[:, t, 0] - prev_seasonal) + \
                         (1 - alpha) * (level[:, t-1] + trend[:, t-1])

            # Обновление тренда
            trend[:, t] = beta * (level[:, t] - level[:, t-1]) + \
                         (1 - beta) * trend[:, t-1]

            # Обновление сезонности
            seasonal[:, t] = gamma * (x[:, t, 0] - level[:, t]) + \
                            (1 - gamma) * prev_seasonal

        # Объединяем компоненты
        components = torch.stack([level, trend, seasonal], dim=-1)
        return components

class ESRNN(nn.Module):
    """ES-RNN модель, комбинирующая экспоненциальное сглаживание с RNN"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2,
                 pred_length=PRED_LENGTH, seasonality_period=7):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pred_length = pred_length

        # Экспоненциальное сглаживание для основного сигнала
        self.es_module = ExponentialSmoothing(seasonality_period)

        # Проекция входных признаков
        self.input_projection = nn.Linear(input_size, hidden_size)

        # RNN для моделирования остатков и контекста
        self.lstm = nn.LSTM(
            input_size=hidden_size + 3,  # hidden_size + 3 ES компоненты
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Внимание для агрегации временных шагов
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        # Глобальный контекст через пулинг
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Комбинирование ES и RNN предсказаний
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size // 2 + 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        # Выходной слой для многошагового прогноза
        self.output_layer = nn.Linear(hidden_size // 2, pred_length)

        # Дополнительный выход для прямого ES прогноза
        self.es_projection = nn.Sequential(
            nn.Linear(3, pred_length),
            nn.Tanh()
        )

        # Веса для комбинирования ES и RNN прогнозов
        self.combination_weights = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Экспоненциальное сглаживание основного сигнала (log returns)
        es_components = self.es_module(x[:, :, 0:1])  # (batch, seq, 3)

        # 2. Проекция всех входных признаков
        x_projected = self.input_projection(x)  # (batch, seq, hidden)

        # 3. Объединение с ES компонентами
        lstm_input = torch.cat([x_projected, es_components], dim=-1)

        # 4. LSTM обработка
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # 5. Self-attention на LSTM выходах
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 6. Глобальный контекст через адаптивный пулинг
        # Переставляем размерности для пулинга: (batch, seq, hidden) -> (batch, hidden, seq)
        global_context = self.global_pool(attn_out.transpose(1, 2))

        # 7. Последние ES компоненты для прогноза
        last_es = es_components[:, -1, :]  # (batch, 3)

        # 8. ES-based прогноз
        es_forecast = self.es_projection(last_es)  # (batch, pred_length)

        # 9. Комбинирование контекстов
        combined_features = torch.cat([global_context, last_es], dim=-1)
        fused = self.fusion_layer(combined_features)

        # 10. RNN-based прогноз
        rnn_forecast = self.output_layer(fused)  # (batch, pred_length)

        # 11. Взвешенная комбинация прогнозов
        weight = torch.sigmoid(self.combination_weights)
        final_forecast = weight * es_forecast + (1 - weight) * rnn_forecast

        return final_forecast

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Устройство: {device}")
    print(f"Количество параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    # Вывод обучаемых параметров ES
    print("\nПараметры экспоненциального сглаживания:")
    print(f"  Alpha (уровень): {torch.sigmoid(model.es_module.alpha).item():.3f}")
    print(f"  Beta (тренд): {torch.sigmoid(model.es_module.beta).item():.3f}")
    print(f"  Gamma (сезонность): {torch.sigmoid(model.es_module.gamma).item():.3f}")

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                          batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_loss = float('inf')
    no_improve = 0

    for epoch in tqdm(range(epochs), desc="Обучение"):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                outputs = model(X_val_batch.to(device))
                val_loss += criterion(outputs, y_val_batch.to(device)).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_esrnn.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")
            # Вывод текущих параметров ES
            print(f"  ES params - α: {torch.sigmoid(model.es_module.alpha).item():.3f}, "
                  f"β: {torch.sigmoid(model.es_module.beta).item():.3f}, "
                  f"γ: {torch.sigmoid(model.es_module.gamma).item():.3f}, "
                  f"weight: {torch.sigmoid(model.combination_weights).item():.3f}")

    model.load_state_dict(torch.load('best_esrnn.pth'))
    return model

def plot_predictions(model, X_test, y_test, test_dates, original_data):
    """Визуализация прогнозов модели с анализом компонент"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        test_input = torch.FloatTensor(X_test).to(device)
        pred_returns = model(test_input).cpu().numpy()

        # Получаем ES компоненты для анализа
        es_components = model.es_module(test_input[:, :, 0:1]).cpu().numpy()

    # Конвертация обратно в цены
    last_prices = []
    for i, date in enumerate(test_dates):
        idx = original_data.index.get_loc(date)
        last_prices.append(original_data['Close'].iloc[idx-1])
    last_prices = np.array(last_prices)

    # Прогнозируемые цены
    pred_prices = np.zeros_like(pred_returns)
    actual_prices = np.zeros_like(y_test)

    for i in range(len(pred_returns)):
        # Прогнозы
        cumulative_returns = np.cumsum(pred_returns[i])
        pred_prices[i] = last_prices[i] * np.exp(cumulative_returns)

        # Фактические
        cumulative_actual = np.cumsum(y_test[i])
        actual_prices[i] = last_prices[i] * np.exp(cumulative_actual)

    # Создание DataFrame для анализа
    results = []
    for i in range(len(test_dates)):
        for j in range(PRED_LENGTH):
            results.append({
                'Date': test_dates[i] + timedelta(days=j+1),
                'Horizon': j+1,
                'Predicted': pred_prices[i, j],
                'Actual': actual_prices[i, j],
                'Predicted_Return': pred_returns[i, j],
                'Actual_Return': y_test[i, j]
            })

    results_df = pd.DataFrame(results)

    # Визуализация
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # График 1: Прогнозы для разных горизонтов
    for horizon in range(1, min(4, PRED_LENGTH+1)):
        horizon_data = results_df[results_df['Horizon'] == horizon]
        axes[0, 0].plot(horizon_data['Date'], horizon_data['Predicted'],
                       label=f'Прогноз {horizon}д', alpha=0.7)
        axes[0, 0].plot(horizon_data['Date'], horizon_data['Actual'],
                       label=f'Факт {horizon}д', alpha=0.7, linestyle='--')

    axes[0, 0].set_title('Прогнозы цен для разных горизонтов')
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Цена ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Ошибки прогнозирования
    errors = results_df.groupby('Horizon').apply(
        lambda x: np.abs(x['Predicted'] - x['Actual']).mean()
    )
    axes[0, 1].bar(errors.index, errors.values)
    axes[0, 1].set_title('Средняя абсолютная ошибка по горизонтам')
    axes[0, 1].set_xlabel('Горизонт прогноза (дни)')
    axes[0, 1].set_ylabel('MAE ($)')
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Scatter plot прогноз vs факт
    for horizon in range(1, min(4, PRED_LENGTH+1)):
        horizon_data = results_df[results_df['Horizon'] == horizon]
        axes[1, 0].scatter(horizon_data['Actual'], horizon_data['Predicted'],
                          alpha=0.5, label=f'{horizon} день')

    # Линия идеального прогноза
    min_price = results_df['Actual'].min()
    max_price = results_df['Actual'].max()
    axes[1, 0].plot([min_price, max_price], [min_price, max_price],
                    'r--', label='Идеальный прогноз')
    axes[1, 0].set_title('Прогноз vs Факт')
    axes[1, 0].set_xlabel('Фактическая цена ($)')
    axes[1, 0].set_ylabel('Прогнозируемая цена ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # График 4: Распределение ошибок
    errors_dist = results_df['Predicted'] - results_df['Actual']
    axes[1, 1].hist(errors_dist, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Нулевая ошибка')
    axes[1, 1].set_title('Распределение ошибок прогнозирования')
    axes[1, 1].set_xlabel('Ошибка ($)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # График 5: ES компоненты (пример для первых 100 точек теста)
    sample_size = min(100, len(es_components))
    sample_indices = np.arange(sample_size)

    for i in sample_indices[:5]:  # Показываем первые 5 примеров
        axes[2, 0].plot(es_components[i, :, 0], alpha=0.5, label=f'Уровень {i}')
    axes[2, 0].set_title('Примеры ES компонент: Уровень')
    axes[2, 0].set_xlabel('Временной шаг')
    axes[2, 0].set_ylabel('Значение')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # График 6: Средние значения ES компонент
    mean_level = np.mean(es_components[:sample_size, :, 0], axis=0)
    mean_trend = np.mean(es_components[:sample_size, :, 1], axis=0)
    mean_seasonal = np.mean(es_components[:sample_size, :, 2], axis=0)

    axes[2, 1].plot(mean_level, label='Уровень', linewidth=2)
    axes[2, 1].plot(mean_trend, label='Тренд', linewidth=2)
    axes[2, 1].plot(mean_seasonal, label='Сезонность', linewidth=2)
    axes[2, 1].set_title('Средние ES компоненты')
    axes[2, 1].set_xlabel('Временной шаг')
    axes[2, 1].set_ylabel('Значение')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Метрики качества
    print("\nМетрики качества прогнозирования:")
    for horizon in range(1, PRED_LENGTH+1):
        horizon_data = results_df[results_df['Horizon'] == horizon]
        mae = np.abs(horizon_data['Predicted'] - horizon_data['Actual']).mean()
        mape = (np.abs((horizon_data['Predicted'] - horizon_data['Actual']) / horizon_data['Actual']) * 100).mean()
        rmse = np.sqrt(((horizon_data['Predicted'] - horizon_data['Actual'])**2).mean())

        print(f"\nГоризонт {horizon} день:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: ${rmse:.2f}")

    # Финальные параметры ES
    print("\nФинальные параметры экспоненциального сглаживания:")
    print(f"  Alpha (уровень): {torch.sigmoid(model.es_module.alpha).item():.3f}")
    print(f"  Beta (тренд): {torch.sigmoid(model.es_module.beta).item():.3f}")
    print(f"  Gamma (сезонность): {torch.sigmoid(model.es_module.gamma).item():.3f}")
    print(f"  Вес ES в финальном прогнозе: {torch.sigmoid(model.combination_weights).item():.3f}")

    return results_df

def main():
    # Загрузка и подготовка данных
    data_scaled, original_data = load_crypto_data()
    dates = original_data.index

    # Создание последовательностей
    X, y, X_dates = create_sequences(data_scaled, dates)

    # Проверка размерностей
    print(f"Форма X: {X.shape}, форма y: {y.shape}")

    # Разделение данных
    test_size = int(len(X) * TEST_SIZE)
    val_size = int(len(X) * VAL_SIZE)
    train_size = len(X) - val_size - test_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test, test_dates = X[train_size+val_size:], y[train_size+val_size:], X_dates[train_size+val_size:]

    # Инициализация модели
    input_size = X_train.shape[2] if len(X_train.shape) > 2 else 1
    model = ESRNN(input_size=input_size, hidden_size=128, num_layers=2, seasonality_period=7)

    # Обучение
    if os.path.exists(MODEL_PATH):
        print(f"Загрузка модели из {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        model = train_model(model, X_train, y_train, X_val, y_val)
        torch.save(model.state_dict(), MODEL_PATH)

    # Прогнозирование и визуализация
    results_df = plot_predictions(model, X_test, y_test, test_dates, original_data)

    # Оценка точности в абсолютных ценах
    test_preds = results_df.groupby('Horizon').apply(
        lambda x: np.mean(np.abs(x['Predicted'] - x['Actual']))
    )
    print("\nСредняя абсолютная ошибка по дням прогноза ($):")
    print(test_preds)

if __name__ == "__main__":
    main()