#include <iostream>
using namespace std;

// Константа для размера доски (8 ферзей)
const int N = 8;

// Переменная для подсчета решений
int totalSolutions = 0;

// Шахматная доска представлена матрицей bool
bool chessBoard[N][N];

// Функция проверки безопасности для конкретного положения ферзя
bool isPositionSafe(int row, int column) {
    // Проверка вертикали (столбца)
    for (int i = 0; i < row; i++) {
        if (chessBoard[i][column])
            return false;
    }

    // Проверка главной диагонали (слева-вверх)
    for (int i = row, j = column; i >= 0 && j >= 0; i--, j--) {
        if (chessBoard[i][j])
            return false;
    }

    // Проверка второй диагонали (справа-вверх)
    for (int i = row, j = column; i >= 0 && j < N; i--, j++) {
        if (chessBoard[i][j])
            return false;
    }

    return true;
}

// Вспомогательная функция для вывода текущего состояния доски
void displayBoard() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << (chessBoard[i][j] ? "Q " : ". ");
        }
        cout << endl;
    }
    cout << endl;
}

// Рекурсивная функция backtracking для поиска решений
void findSolution(int currentRow) {
    if (currentRow == N) { // Если добрались до конца доски, нашли решение
        totalSolutions++;
        displayBoard(); // Вывести текущее решение
        return;
    }

    // Пробуем поставить ферзя в каждом столбце текущей строки
    for (int col = 0; col < N; col++) {
        if (isPositionSafe(currentRow, col)) { // Если позиция безопасна
            chessBoard[currentRow][col] = true; // Поставить ферзя
            findSolution(currentRow + 1);       // Продолжить на следующей строке
            chessBoard[currentRow][col] = false;// Вернуть позицию (backtracking)
        }
    }
}

// Главная функция
int main() {
    // Инициализируем всю доску как пустую
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            chessBoard[i][j] = false;
        }
    }

    // Запустить поиск решений с первой строки
    findSolution(0);

    // Показать общее количество решений
    cout << "Всего решений: " << totalSolutions << endl;

    return 0;
}