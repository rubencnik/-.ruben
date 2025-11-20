import java.util.ArrayList;
import java.util.List;

public class TSPNearestNeighbor {
    
    public static List<Integer> tspNearest(int[][] dist) {
        // Шаг 1: Инициализация структур данных
        List<Integer> path = new ArrayList<>();  // Список для хранения маршрута
        boolean[] visited = new boolean[dist.length];  // Массив для отслеживания посещённых городов
        int curr = 0;  // Начинаем с города 0
        
        // Шаг 2: Добавляем стартовый город в маршрут
        path.add(curr);
        visited[curr] = true;  // Помечаем город 0 как посещённый
        
        // Шаг 3: Основной цикл - посещаем все оставшиеся города
        while (path.size() < dist.length) {
            int nextCity = -1;  // Переменная для хранения следующего города
            int minDist = Integer.MAX_VALUE;  // Переменная для хранения минимального расстояния
            
            // Шаг 4: Поиск ближайшего непосещённого города
            for (int i = 0; i < dist.length; i++) {
                // Проверяем: город не посещён И расстояние меньше текущего минимума И расстояние > 0 (не тот же город)
                if (!visited[i] && dist[curr][i] < minDist && dist[curr][i] > 0) {
                    minDist = dist[curr][i];  // Обновляем минимальное расстояние
                    nextCity = i;  // Запоминаем город с минимальным расстоянием
                }
            }
            
            // Шаг 5: Добавление найденного города в маршрут
            if (nextCity != -1) {
                path.add(nextCity);  // Добавляем ближайший город в маршрут
                visited[nextCity] = true;  // Помечаем его как посещённый
                curr = nextCity;  // Перемещаемся в этот город для следующей итерации
            } else {
                // Шаг 6: Защита от ошибок - если нет доступных городов
                break;
            }
        }
        
        return path;  // Возвращаем построенный маршрут
    }
    
    // Метод для вычисления общей длины пути
    public static int calculateTotalDistance(List<Integer> path, int[][] dist) {
        // Шаг 1: Инициализация переменной для общей длины
        int totalDistance = 0;
        
        // Шаг 2: Суммируем расстояния между последовательными городами в маршруте
        for (int i = 0; i < path.size() - 1; i++) {
            int from = path.get(i);      // Текущий город
            int to = path.get(i + 1);    // Следующий город
            totalDistance += dist[from][to];  // Добавляем расстояние между ними
        }
        
        return totalDistance;  // Возвращаем общую длину маршрута
    }
    
    // Пример использования
    public static void main(String[] args) {
        // Шаг 1: Создаём матрицу расстояний между 5 городами
        int[][] distances = {
            // Расстояния от каждого города до других:
            {0, 10, 15, 20, 30},  // Из города 0: до 0,1,2,3,4
            {10, 0, 35, 25, 40},   // Из города 1: до 0,1,2,3,4
            {15, 35, 0, 30, 50},   // Из города 2: до 0,1,2,3,4
            {20, 25, 30, 0, 45},   // Из города 3: до 0,1,2,3,4
            {30, 40, 50, 45, 0}    // Из города 4: до 0,1,2,3,4
        };
        
        // Шаг 2: Находим маршрут алгоритмом ближайшего соседа
        List<Integer> route = tspNearest(distances);
        
        // Шаг 3: Выводим результат
        System.out.println("Маршрут коммивояжёра (алгоритм ближайшего соседа):");
        for (int i = 0; i < route.size(); i++) {
            System.out.print("Город " + route.get(i));
            if (i < route.size() - 1) {
                System.out.print(" > ");  // Стрелка между городами
            }
        }
        
        // Шаг 4: Вычисляем и выводим общую длину пути
        int totalDist = calculateTotalDistance(route, distances);
        System.out.println("\nОбщая длина пути: " + totalDist);
        
        // Шаг 5: Демонстрация работы на другом примере
        System.out.println("\n--- Дополнительный пример с 4 городами ---");
        int[][] distances2 = {
            {0, 2, 9, 10},   // Город 0
            {1, 0, 6, 4},    // Город 1
            {15, 7, 0, 8},   // Город 2
            {6, 3, 12, 0}    // Город 3
        };
        
        List<Integer> route2 = tspNearest(distances2);
        System.out.println("Маршрут для 4 городов:");
        for (int i = 0; i < route2.size(); i++) {
            System.out.print("Город " + route2.get(i));
            if (i < route2.size() - 1) {
                System.out.print(" > ");
            }
        }
        System.out.println("\nОбщая длина пути: " + calculateTotalDistance(route2, distances2));
        
        // Шаг 6: Подробное объяснение работы на примере
        System.out.println("\n--- Пошаговое объяснение для 4 городов ---");
        System.out.println("1. Начинаем с города 0");
        System.out.println("2. Из города 0 ищем ближайший непосещённый:");
        System.out.println("   - До города 1: расстояние 2 (минимум)");
        System.out.println("   - До города 2: расстояние 9");
        System.out.println("   - До города 3: расстояние 10");
        System.out.println("3. Переходим в город 1");
        System.out.println("4. Из города 1 ищем ближайший непосещённый:");
        System.out.println("   - До города 2: расстояние 6");
        System.out.println("   - До города 3: расстояние 4 (минимум)");
        System.out.println("5. Переходим в город 3");
        System.out.println("6. Из города 3 остался только город 2: расстояние 12");
        System.out.println("7. Переходим в город 2");
        System.out.println("8. Все города посещены - маршрут построен!");
    }
}