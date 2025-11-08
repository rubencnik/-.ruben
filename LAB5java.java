import java.util.Arrays;

// Изменение имени класса на LAB5Java
public class LAB5Java {

    // Переменная для подсчета решений
    private static int solutionCount = 0;

    public static void main(String[] args) {
        int N = 8; // размер доски (количество ферзей)
        boolean[][] board = new boolean[N][N]; // начальная доска, false - свободно, true - занято

        placeQueen(board, 0); // начинаем размещать ферзей с первой строки
        System.out.println("Общее количество решений: " + solutionCount);
    }

    /**
     * Метод решает задачу раскладки ферзей, используя backtracking.
     *
     * @param board - шахматная доска
     * @param row   - номер текущей строки
     */
    private static void placeQueen(boolean[][] board, int row) {
        int N = board.length;

        // Если дошли до конца доски, нашли решение
        if (row == N) {
            solutionCount++;
            printBoard(board);
            return;
        }

        // Попытка установить ферзя в каждом столбце текущей строки
        for (int col = 0; col < N; col++) {
            if (isSafe(board, row, col)) {
                board[row][col] = true;      // ставим ферзя
                placeQueen(board, row + 1);  // переходим к следующей строке
                board[row][col] = false;     // откатываем изменение (backtracking)
            }
        }
    }

    /**
     * Проверяет, безопасно ли ставить ферзя в заданную позицию.
     *
     * @param board - шахматная доска
     * @param row   - номер строки
     * @param col   - номер столбца
     * @return true, если позиция безопасна, иначе false
     */
    private static boolean isSafe(boolean[][] board, int row, int col) {
        int N = board.length;

        // Проверка вертикали (над ферзем)
        for (int i = 0; i < row; i++) {
            if (board[i][col]) {
                return false;
            }
        }

        // Проверка главной диагонали вправо-вниз
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j]) {
                return false;
            }
        }

        // Проверка дополнительной диагонали вправо-вверх
        for (int i = row, j = col; i >= 0 && j < N; i--, j++) {
            if (board[i][j]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Выводит текущее состояние доски.
     *
     * @param board - шахматная доска
     */
    private static void printBoard(boolean[][] board) {
        int N = board.length;
        for (boolean[] row : board) {
            StringBuilder sb = new StringBuilder();
            for (boolean cell : row) {
                sb.append(cell ? "Q " : ". ");
            }
            System.out.println(sb.toString());
        }
        System.out.println(); // пропуск строки между решениями
    }
}