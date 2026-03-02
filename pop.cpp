#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

bool isPointInArea(double x, double y);  // Проверка области рисунка
void printPointInfo(double x, double y); // Анализ точки

int main() {
    double x, y;
    
    cout << fixed << setprecision(2);  // Формат вывода
    cout << "=== ВАРИАНТ 3: Ветвления ===\n";
    cout << "Принадлежность точки (x,y) заштрихованной области\n";
    cout << "0 0 для выхода\n\n";
    
    // Цикл для любого количества точек
    while (true) {
        cout << "Введите x, y: ";
        cin >> x >> y;
        
        // Условие выхода
        if (x == 0.0 && y == 0.0) {
            break;
        }
        
        printPointInfo(x, y);  // Подробный анализ точки
        cout << endl;
    }
    
    cout << "\nПрограмма выполнена!" << endl;
    return 0;
}

// Вывод подробной информации о точке
void printPointInfo(double x, double y) {
    cout << "Точка (" << setw(6) << x << ", " << setw(6) << y << "):" << endl;
    
    // Анализ знака и диапазона X
    cout << "  X: ";
    if (x < 0) cout << "отрицательный";
    else if (x > 0) cout << "положительный";
    else cout << "ноль";
    cout << " (";
    if (x >= -5.0 && x <= 5.0) cout << "[-5..5]";
    else cout << "вне [-5..5]";
    cout << ")" << endl;
    
    // Анализ знака и диапазона Y
    cout << "  Y: ";
    if (y < 0) cout << "отрицательный";
    else if (y > 0) cout << "положительный";
    else cout << "ноль";
    cout << " (";
    if (y >= -8.0 && y <= 3.0) cout << "[-8..3]";
    else cout << "вне [-8..3]";
    cout << ")" << endl;
    
    // Проверка принадлежности области
    cout << "  Область: ";
    if (isPointInArea(x, y)) {
        cout << "✓ В ОБЛАСТИ" << endl;
    } else {
        cout << "✗ ВНЕ ОБЛАСТИ" << endl;
    }
}

// Логика заштрихованной области
bool isPointInArea(double x, double y) {
    bool left = (x >= -5.0 && x <= 0.0) &&      // Диапазон X левого
                (y >= -1.0 * x - 3.0) &&        // Нижняя граница
                (y <=  1.0 * x + 3.0);          // Верхняя граница
    
    bool right = (x >= 0.0 && x <= 5.0) &&       // Диапазон X правого
                 (y >=  1.0 * x - 3.0) &&       // Нижняя граница
                 (y <= -1.0 * x + 3.0);         // Верхняя граница
    
    bool center = (fabs(x) <= 1.0) &&            // Центр по X
                  (y >= 0.0 && y <= 3.0);       // Центр по Y
    
    return left || right || center;              // Объединение областей
}
