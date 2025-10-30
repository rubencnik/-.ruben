"""
�������(��������� ����������)
"""
def insertion_sort(bucket):
    """������������ ��� ���������� ���������� ����."""
    for i in range(1, len(bucket)):
        up = bucket[i]
        j = i - 1
        while j >= 0 and bucket[j] > up:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = up
    return bucket

def bucket_sort(arr):
    """�������� ������� ������� ����������."""
    if not arr or len(arr) <= 1:
        return arr
    
    # ��������� ������������ � ����������� ��������
    max_value = max(arr)
    min_value = min(arr)
    
    # ����� ����� � ������ ��������� ��� ������� ����
    bucket_count = len(arr)
    bucket_size = (max_value - min_value) / bucket_count
    
    # �������� ������ �����
    buckets = [[] for _ in range(bucket_count)]
    
    # ������������� ��������� �� �����
    for value in arr:
        # �������� ���������� ��� ��� ������� ��������
        index = int((value - min_value) // bucket_size)
        if index != bucket_count:
            buckets[index].append(value)
        else:
            buckets[-1].append(value)  # ������� ������, ��������� ������� �������� � ��������� ���
    
    # ���������� ������� ����
    final_result = []
    for bucket in buckets:
        final_result.extend(insertion_sort(bucket))
    
    return final_result

# ������������
numbers = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
sorted_numbers = bucket_sort(numbers)
print(sorted_numbers)
"""
�����:

[0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
"""



"""
������� ����������
"""
def flip(arr, k):
    """����������� ������ k ��������� �������"""
    left = 0
    right = k - 1
    while left < right:
        # ������ ������� ����� � ������ ��������
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

def pancake_sort(arr):
    """������������ ������� ���������� �������"""
    current_size = len(arr)
    
    # ���������� ����������, ���� �� ��������� ���� �������
    while current_size > 1:
        # ������� ������ ����������� �������� � ������� ��������������� �����
        max_index = arr.index(max(arr[:current_size]))
        
        # ���� �������� ��� �� �������, ���������� ��� ������� ����� �������
        if max_index != current_size - 1:
            # ��������� ������� ����� �������, ����� �������� ����� �� �������
            flip(arr, max_index + 1)
            
            # ������ �������������� ��� ������� ����� �������, ����� �������� ��������� ����
            flip(arr, current_size)
        
        # ��������� ������ ��������������� �����
        current_size -= 1
    
    return arr

# ������ �������������
arr = [3, 6, 2, 4, 5, 1]
pancake_sorted_arr = pancake_sort(arr)
print(pancake_sorted_arr)

"""
�����:

[1, 2, 3, 4, 5, 6]
"""

"""
���������� ��������
"""
def bead_sort(arr):
    """������������ ���������� ��������"""
    # �������� ������������ ������ �������
    max_height = max(arr)
    
    # ��������� ������� �������: True �������� ������� �������
    beads_matrix = [[False]*len(arr) for _ in range(max_height)]
    
    # ���������� ������� ���������
    for col, height in enumerate(arr):
        # ������������� ������� � ��������������� ������
        for row in range(height):
            beads_matrix[row][col] = True
    
    # "��������� �������� ������": �������� �� �� ����� �����-�����
    for row in range(max_height):
        count = sum(beads_matrix[row])  # ������� ���������� ������� � ����
        # ����������� ������� �����-�������
        beads_matrix[row] = [True]*count + [False]*(len(arr)-count)
    
    # ��������������� ��������������� ������ �� �������
    sorted_arr = []
    for col in range(len(arr)):
        # ������� ������ ������ �������
        height = sum([beads_matrix[row][col] for row in range(max_height)])
        sorted_arr.append(height)
    
    return sorted_arr

# ������ �������������
arr = [5, 3, 1, 7, 4]
sorted_arr = bead_sort(arr)
print(sorted_arr)

"""
�����:
[7, 5, 4, 3, 1]
"""

"""
����� ��������
"""
import math

def jump_search(arr, target):
    """������������ ����� ������� � ��������������� �������"""
    n = len(arr)
    
    # ���������� ����������� ��� (���������� ������ �� ����� �������)
    step = int(math.sqrt(n))
    
    # ��������� ����� ������
    prev = 0
    
    # ������� ����, ���������� ������� �������
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return None  # ������� ������� �����������
    
    # �������� ����� � ��������� �����
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return None  # ������� ������� �����������
    
    # ���������, ��������� �� ������� ������� � �����
    if arr[prev] == target:
        return prev  # ������� ������� ���������� ��������
    
    return None  # ������� ������� �����������

# ������ �������������
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
target = 55
result = jump_search(arr, target)
print(f'������� {target} ������ �� �������: {result}')

"""
�����:

������� 55 ������ �� �������: 10
"""


"""
���������������� ����� (Exponential Search)
"""
def binary_search(arr, low, high, x):
    """�������� ����� ������ ��������� ���������"""
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # ������� �� ������

def exponential_search(arr, x):
    """������������ ���������������� ����� � ��������������� �������"""
    n = len(arr)
    
    # ���� ������ ������ ��� ���� ������ ������� ��������
    if n == 0 or x < arr[0]:
        return -1
    
    # ���� ��������� ����� ��� ��������� ������
    i = 1
    while i < n and arr[i] <= x:
        i *= 2  # ��������������� ����������� ������� ������
    
    # ������� ��� ��������� ������
    low = i // 2
    high = min(i, n - 1)
    
    # ��������� �������� ����� � ������������ �������
    return binary_search(arr, low, high, x)

# ������ �������������
arr = [2, 3, 4, 10, 40, 50, 60, 70, 80, 90, 100]
target = 10
result = exponential_search(arr, target)
if result != -1:
    print(f'������� {target} ������ �� �������: {result}')
else:
    print('������� �� ������')

"""
�����:

������� 10 ������ �� �������: 3
"""