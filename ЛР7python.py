def greedy_knapsack_2approx(items, capacity):
    """
    2-аппроксимационный жадный алгоритм для 0-1 рюкзака.
    """
    # Сортируем по убыванию удельной стоимости (value/weight)
    sorted_items = sorted(
        items,
        key=lambda x: x['value'] / x['weight'],
        reverse=True
    )
    
    # Жадный проход: добавляем по порядку, если помещается
    selected_greedy = []
    total_weight = 0
    total_value_greedy = 0
    
    for item in sorted_items:
        if total_weight + item['weight'] <= capacity:
            selected_greedy.append(item)
            total_weight += item['weight']
            total_value_greedy += item['value']

    # Находим самый ценный отдельный предмет (который помещается)
    best_single = None
    best_single_value = 0
    for item in items:
        if item['weight'] <= capacity and item['value'] > best_single_value:
            best_single = item
            best_single_value = item['value']

    # Выбираем лучшее: жадное решение или один предмет
    if total_value_greedy >= best_single_value:
        selected = selected_greedy
        total_value = total_value_greedy
    else:
        selected = [best_single]
        total_value = best_single_value

    approx_ratio = 0.5  # гарантированная нижняя граница

    return selected, total_value, approx_ratio



def main():
    print("=== 2-аппроксимационный алгоритм для задачи 0-1 рюкзак ===\n")
    
    # Ввод количества предметов
    try:
        n = int(input("Введите количество предметов: "))
        if n <= 0:
            print("Количество предметов должно быть положительным числом.")
            return
    except ValueError:
        print("Ошибка: введите целое число.")
        return

    items = []
    
    # Ввод данных о каждом предмете
    print("\nВведите данные о предметах (вес и стоимость):")
    for i in range(n):
        try:
            weight = float(input(f"Предмет {i+1}. Вес: "))
            value = float(input(f"Предмет {i+1}. Стоимость: "))
            
            if weight <= 0 or value < 0:
                print("Вес должен быть положительным, стоимость — неотрицательной.")
                return
                
            items.append({'weight': weight, 'value': value})
        except ValueError:
            print("Ошибка: введите числовое значение.")
            return

    # Ввод вместимости рюкзака
    try:
        capacity = float(input("\nВведите вместимость рюкзака: "))
        if capacity <= 0:
            print("Вместимость рюкзака должна быть положительным числом.")
            return
    except ValueError:
        print("Ошибка: введите числовое значение.")
        return

    # Запуск алгоритма
    selected, total_value, approx_ratio = greedy_knapsack_2approx(items, capacity)

    # Вывод результата
    print("РЕЗУЛЬТАТ:")
    print("="*40)
    print("Выбранные предметы:")
    for item in selected:
        print(f"  Вес: {item['weight']}, Стоимость: {item['value']}")
    print(f"Общая стоимость: {total_value}")
    print(f"Коэффициент аппроксимации : {approx_ratio}")
    print(f"Коэффициент аппроксимации (гарантированный): ≥ {approx_ratio:.1f}")



# Запуск программы
if __name__ == "__main__":
    main()
"""
output:

=== 2-аппроксимационный алгоритм для задачи 0-1 рюкзак ===

Введите количество предметов: 4

Введите данные о предметах (вес и стоимость):
Предмет 1. Вес: 3
Предмет 1. Стоимость: 12
Предмет 2. Вес: 2
Предмет 2. Стоимость: 8
Предмет 3. Вес: 4
Предмет 3. Стоимость: 20
Предмет 4. Вес: 5
Предмет 4. Стоимость: 18

Введите вместимость рюкзака: 7
РЕЗУЛЬТАТ:
========================================
Выбранные предметы:
  Вес: 4.0, Стоимость: 20.0
  Вес: 3.0, Стоимость: 12.0
Общая стоимость: 32.0
Коэффициент аппроксимации (гарантированный): ≥ 0.5
"""