#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

bool isPointInArea(double x, double y);  // Проверка области
void printPointInfo(double x, double y); // Подробная информация о точке

int main() {
    double x, y;
    
    cout << fixed << setprecision(2);
    cout << "=== ВАРИАНТ 3: Ветвления ===\n";
    cout << "Принадлежность точки (x,y) заштрихованной области\n";
    cout << "0 0 для выхода\n\n";
    
    // Универсальный ввод любых точек
    while (true) {
        cout << "Введите x, y: ";
        cin >> x >> y;
        
        if (x == 0.0 && y == 0.0) {
            cout << "Выход.\n\n";
            break;
        }
        
        printPointInfo(x, y);  // Вывод подробной информации
        cout << endl;
    }
    
    // 3 примера с подробной информацией
    cout << "=== 3 ПРИМЕРА С АНАЛИЗОМ ===\n\n";
    
    cout << "ПРИМЕР 1:\n";
    printPointInfo(-2.0, 1.0);
    cout << endl;
    
    cout << "ПРИМЕР 2:\n";
    printPointInfo(2.0, 1.0);
    cout << endl;
    
    cout << "ПРИМЕР 3:\n";  
    printPointInfo(0.0, 3.5);
    
    cout << "\nПрограмма выполнена!" << endl;
    return 0;
}

// Подробная информация о точке
void printPointInfo(double x, double y) {
    cout << "Точка (" << setw(6) << x << ", " << setw(6) << y << "):" << endl;
    
    // Знак X
    cout << "  X: ";
    if (x < 0) cout << "отрицательный";
    else if (x > 0) cout << "положительный";
    else cout << "ноль";
    cout << " (диапазон: ";
    if (x >= -5.0 && x <= 5.0) cout << "[-5..5]";
    else cout << "вне [-5..5]";
    cout << ")" << endl;
    
    // Знак Y
    cout << "  Y: ";
    if (y < 0) cout << "отрицательный";
    else if (y > 0) cout << "положительный";
    else cout << "ноль";
    cout << " (диапазон: ";
    if (y >= -8.0 && y <= 3.0) cout << "[-8..3]";
    else cout << "вне [-8..3]";
    cout << ")" << endl;
    
    // Принадлежность области
    cout << "  Область: ";
    if (isPointInArea(x, y)) {
        cout << "✓ В ОБЛАСТИ" << endl;
    } else {
        cout << "✗ ВНЕ ОБЛАСТИ" << endl;
    }
}

// Логика области (коэффициент 1.0)
bool isPointInArea(double x, double y) {
    bool left = (x >= -5.0 && x <= 0.0) &&      // Левый треугольник по X
                (y >= -1.0 * x - 3.0) &&        // Нижняя граница левого
                (y <=  1.0 * x + 3.0);          // Верхняя граница левого
    
    bool right = (x >= 0.0 && x <= 5.0) &&       // Правый треугольник по X
                 (y >=  1.0 * x - 3.0) &&       // Нижняя граница правого
                 (y <= -1.0 * x + 3.0);         // Верхняя граница правого
    
    bool center = (fabs(x) <= 1.0) &&            // Центр по X
                  (y >= 0.0 && y <= 3.0);       // Центр по Y
    
    return left || right || center;
}
