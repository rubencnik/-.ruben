#include <iostream>
#include <iomanip>
using namespace std;

// Прототип функции скалярного произведения
double scalarProduct(double x1, double y1, double z1, double x2, double y2, double z2);

// Функция ввода координат вектора
void inputVector(double& x, double& y, double& z);

int main() {
    double x1, y1, z1, x2, y2, z2;  // Координаты двух векторов
    double prod1, prod2, prod3;     // Три скалярных произведения
    double average;                 // Среднее арифметическое
    
    cout << fixed << setprecision(2);
    cout << "=== ВАРИАНТ 3: Циклы + Функции ===\n";
    cout << "Скалярное произведение векторов в 3D\n\n";
    
    // ★★★ ПЕРВЫЕ ДВА ВЕКТОРА ★★★
    cout << "=== ПАРА 1 ===" << endl;
    cout << "Вектор A: "; inputVector(x1, y1, z1);  // Ввод вектора A
    cout << "Вектор B: "; inputVector(x2, y2, z2);  // Ввод вектора B
    
    prod1 = scalarProduct(x1, y1, z1, x2, y2, z2);  // Вычисление скалярного произведения
    cout << "A·B = " << setw(8) << prod1 << endl << endl;
    
    // ★★★ ВТОРАЯ ПАРА ВЕКТОРОВ ★★★
    cout << "=== ПАРА 2 ===" << endl;
    cout << "Вектор C: "; inputVector(x1, y1, z1);
    cout << "Вектор D: "; inputVector(x2, y2, z2);
    
    prod2 = scalarProduct(x1, y1, z1, x2, y2, z2);
    cout << "C·D = " << setw(8) << prod2 << endl << endl;
    
    // ★★★ ТРЕТЬЯ ПАРА ВЕКТОРОВ ★★★
    cout << "=== ПАРА 3 ===" << endl;
    cout << "Вектор E: "; inputVector(x1, y1, z1);
    cout << "Вектор F: "; inputVector(x2, y2, z2);
    
    prod3 = scalarProduct(x1, y1, z1, x2, y2, z2);
    cout << "E·F = " << setw(8) << prod3 << endl << endl;
    
    // Среднее арифметическое трех скалярных произведений
    average = (prod1 + prod2 + prod3) / 3.0;
    cout << "=== РЕЗУЛЬТАТ ===" << endl;
    cout << "Среднее: " << setw(8) << average << endl;
    
    cout << "\nПрограмма выполнена!" << endl;
    return 0;
}

// Функция скалярного произведения двух векторов A·B = x1*x2 + y1*y2 + z1*z2
double scalarProduct(double x1, double y1, double z1, double x2, double y2, double z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;  // Формула скалярного произведения
}

// Функция ввода координат вектора (передача по ссылке для изменения значений)
void inputVector(double& x, double& y, double& z) {
    cin >> x >> y >> z;  // Считывание трех координат
}
