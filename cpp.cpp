#include <iostream>     // cin, cout
#include <iomanip>      // setw, setprecision
#include <fstream>      // ifstream, ofstream
#include <cmath>        // pow, sin, log2, fabs
#include <string>       // string для чтения строк
using namespace std;

double calculateY(double A, double B, double C);  // Прототип функции

int main() {
    double A, B, C;     // Входные параметры
    double Y;           // Результат
    
    cout << fixed << setprecision(6);  // 6 знаков после запятой
    
    cout << "=== ВАРИАНТ 3: Ввод-вывод данных ===\n\n";
    
    // ★★★ ВАРИАНТ 1: КОНСОЛЬНЫЙ ВВОД ★★★
    cout << "=== ВАРИАНТ 1: КОНСОЛЬ ===" << endl;
    cout << "Введите A, B, C: ";        // Запрос данных
    cin >> A >> B >> C;                // Читаем 3 числа
    
    cout << "\nВведенные данные:\n";
    cout << "A = " << setw(8) << A << endl;  // Выравнивание по 8 символам
    cout << "B = " << setw(8) << B << endl;
    cout << "C = " << setw(8) << C << endl;
    
    Y = calculateY(A, B, C);           // Вычисляем Y
    cout << "\nY = " << setw(12) << Y << endl << endl;
    
    // ★★★ ВАРИАНТ 2: ФАЙЛЫ ★★★
    cout << "=== ВАРИАНТ 2: ФАЙЛЫ ===" << endl;
    
    // Записываем тестовые данные
    ofstream fout("input.txt");        // Создаем файл для записи
    if (!fout) {
        cerr << "Ошибка input.txt!" << endl;
        return 1;                      // Выход при ошибке
    }
    fout << "2.0\n5.0\n3.0\n";         // Тестовые значения
    fout.close();                      // Закрываем файл
    
    cout << "✓ input.txt создан\n";
    
    // Читаем из файла
    ifstream fin("input.txt");         // Открываем для чтения
    if (!fin) {
        cerr << "Ошибка чтения input.txt!" << endl;
        return 1;
    }
    fin >> A >> B >> C;                // Читаем A, B, C из файла
    fin.close();
    
    cout << "\nИз input.txt:\n";
    cout << "A = " << setw(8) << A << endl;
    cout << "B = " << setw(8) << B << endl;
    cout << "C = " << setw(8) << C << endl;
    
    Y = calculateY(A, B, C);           // Пересчитываем
    cout << "Y = " << setw(12) << Y << endl;
    
    // Записываем результат
    ofstream fout_res("output.txt");
    if (!fout_res) {
        cerr << "Ошибка output.txt!" << endl;
        return 1;
    }
    fout_res << "=== ВАРИАНТ 3 ===\n";
    fout_res << "A=" << A << " B=" << B << " C=" << C << endl;
    fout_res << "Y=" << fixed << setprecision(6) << Y << endl;
    fout_res.close();
    
    cout << "\n✓ output.txt готов\n";
    
    // Показываем результат из файла
    cout << "\noutput.txt:\n";
    ifstream fin_res("output.txt");
    string line;
    while (getline(fin_res, line)) {   // Читаем построчно
        cout << line << endl;
    }
    fin_res.close();
    
    cout << "\nГотово!" << endl;
    return 0;
}

// Вычисление по формуле: [(A²+B)/(B*A)]³ + log₂(|A/(C*sin(A))|)
double calculateY(double A, double B, double C) {
    double frac1 = (pow(A, 2) + B) / (B * A);  // Первая дробь
    double part1 = pow(frac1, 3);              // [дробь]³
    
    double sinA = sin(A);
    if (sinA == 0.0) return part1;             // Защита от /0
    
    double frac2 = fabs(A / (C * sinA));       // |A/(C*sin(A))|
    double part2 = log2(frac2);                // log₂
    
    return part1 + part2;                      // part1 + part2
}