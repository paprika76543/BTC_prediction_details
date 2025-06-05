#include <iostream.h>  // Turbo C++ использует .h
#include <math.h>      // Для tanh(), sin(), cos()
#include <stdlib.h>     // Для rand(), srand()
#include <time.h>       // Для clock()
#include <conio.h>      // Для getch()

// Активационная функция (гиперболический тангенс)
float tanh_activation(float x) {
    return tanh(x);
}

// Производная tanh
float tanh_derivative(float x) {
    return 1.0f - x * x;
}

// Класс RNN (без vector, только массивы)
class RNN {
private:
    int input_size;    // Размер входных данных
    int hidden_size;   // Размер скрытого слоя
    int output_size;   // Размер выходных данных

    // Веса и смещения (статические массивы)
    float *Wxh;  // Веса вход -> скрытый слой
    float *Whh;  // Веса скрытый слой -> скрытый слой
    float *Why;  // Веса скрытый слой -> выход
    float *bh;   // Смещение скрытого слоя
    float *by;   // Смещение выходного слоя

    float *h_prev;  // Предыдущее состояние скрытого слоя

public:
    // Конструктор
    RNN(int input_dim, int hidden_dim, int output_dim) {
        input_size = input_dim;
        hidden_size = hidden_dim;
        output_size = output_dim;

        // Выделяем память под массивы
        Wxh = new float[hidden_size * input_size];
        Whh = new float[hidden_size * hidden_size];
        Why = new float[output_size * hidden_size];
        bh = new float[hidden_size];
        by = new float[output_size];
        h_prev = new float[hidden_size];

        // Инициализация нулями
        for (int i = 0; i < hidden_size; i++) h_prev[i] = 0.0f;

        // Инициализация весов случайными числами [-0.5, 0.5]
        srand(time(NULL));
        for (int i = 0; i < hidden_size * input_size; i++) 
            Wxh[i] = (float)rand() / RAND_MAX - 0.5f;
        
        for (int i = 0; i < hidden_size * hidden_size; i++) 
            Whh[i] = (float)rand() / RAND_MAX - 0.5f;
        
        for (int i = 0; i < output_size * hidden_size; i++) 
            Why[i] = (float)rand() / RAND_MAX - 0.5f;
        
        for (int i = 0; i < hidden_size; i++) 
            bh[i] = (float)rand() / RAND_MAX - 0.5f;
        
        for (int i = 0; i < output_size; i++) 
            by[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Прямой проход (предсказание)
    void forward(const float *x, float *y) {
        float *h = new float[hidden_size];

        // Вычисляем новое состояние скрытого слоя
        for (int i = 0; i < hidden_size; i++) {
            h[i] = 0.0f;

            // Вход -> скрытый слой
            for (int j = 0; j < input_size; j++) 
                h[i] += Wxh[i * input_size + j] * x[j];

            // Скрытый слой -> скрытый слой
            for (int j = 0; j < hidden_size; j++) 
                h[i] += Whh[i * hidden_size + j] * h_prev[j];

            // Активация
            h[i] = tanh_activation(h[i] + bh[i]);
        }

        // Вычисляем выход
        for (int i = 0; i < output_size; i++) {
            y[i] = 0.0f;
            for (int j = 0; j < hidden_size; j++) 
                y[i] += Why[i * hidden_size + j] * h[j];
            y[i] += by[i];
        }

        // Обновляем состояние
        for (int i = 0; i < hidden_size; i++) 
            h_prev[i] = h[i];

        delete[] h;
    }

    // Обучение RNN
    void train(float inputs[][2], float targets[][2], int seq_length, int epochs, float learning_rate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;

            for (int t = 0; t < seq_length; t++) {
                float y_pred[2];
                forward(inputs[t], y_pred);

                // Вычисляем ошибку (MSE)
                float loss = 0.0f;
                for (int i = 0; i < output_size; i++) {
                    float error = y_pred[i] - targets[t][i];
                    loss += 0.5f * error * error;
                }
                total_loss += loss;

                // Упрощённый обратный проход (BPTT)
                for (int i = 0; i < output_size; i++) {
                    float error = y_pred[i] - targets[t][i];
                    for (int j = 0; j < hidden_size; j++) {
                        Why[i * hidden_size + j] -= learning_rate * error * h_prev[j];
                    }
                    by[i] -= learning_rate * error;
                }
            }

            if (epoch % 10 == 0) {
                cout << "Epoch " << epoch << ", Loss: " << total_loss << endl;
            }
        }
    }

    // Деструктор (освобождаем память)
    ~RNN() {
        delete[] Wxh;
        delete[] Whh;
        delete[] Why;
        delete[] bh;
        delete[] by;
        delete[] h_prev;
    }
};

int main() {
    clrscr();  // Очистка экрана (из conio.h)

    // 1. Подготовка данных: синус и косинус
    const int seq_length = 100;
    float inputs[seq_length][2];
    float targets[seq_length][2];

    for (int t = 0; t < seq_length; t++) {
        inputs[t][0] = sin(t * 0.1f);
        inputs[t][1] = cos(t * 0.1f);
        targets[t][0] = sin((t + 1) * 0.1f);
        targets[t][1] = cos((t + 1) * 0.1f);
    }

    // Замер времени начала
    clock_t start = clock();

    // 2. Создаём RNN
    RNN rnn(2, 32, 2);  // 2 входа, 32 нейрона, 2 выхода

    // 3. Обучаем сеть
    rnn.train(inputs, targets, seq_length, 100, 0.01f);

    // Замер времени конца
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "Training time: " << duration << " seconds" << endl;
    cout << "Press any key to exit..." << endl;
    getch();  // Ожидание нажатия клавиши

    return 0;
}
