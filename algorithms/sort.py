import random


# Find minimum elem and put at beginning
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i

        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


# Compare elem to next elem, swap if it's bigger
# After each round last elem is sorted
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - 1 - i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


# Insert elem to previous sorted array
# 0...i-1 is sorted, at i compare it with i-1 if v[i-1] > v[i], shifting
# arr[j] = arr[j-1] e.g 1 4 5 7 8 -> 3  1 3 4 5 7 8
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        j = i
        key = arr[i]
        while j >= 1 and key < arr[j-1]:
            arr[j] = arr[j-1]
            j -= 1
        arr[j] = key

    return arr


# Merge sort, divide and conquer, split array into 2 half and sort each
# merge 2 sorted arrays into final array
def merge(arr, l, m, r):
    le = arr[l:m+1]
    ri = arr[m+1:r+1]

    i, j = 0, 0
    k = l
    while i < len(le) and j < len(ri):
        if le[i] <= ri[j]:
            arr[k] = le[i]
            i += 1
        else:
            arr[k] = ri[j]
            j += 1
        k += 1

    while i < len(le):
        arr[k] = le[i]
        i += 1
        k += 1

    while j < len(ri):
        arr[k] = ri[j]
        j += 1
        k += 1


def merge_sort(arr, l, r):
    if l < r:
        m = (l + r)/2

        merge_sort(arr, l, m)
        merge_sort(arr, m+1, r)
        merge(arr, l, m, r)
        return arr


# Heap sort, build heap first
# swap first element to last
# do heapify from i
def heapify(arr, n, i):
    l = i*2 + 1
    r = i*2 + 2

    _max = i
    if l < n and arr[_max] < arr[l]:
        _max = l

    if r < n and arr[_max] < arr[r]:
        _max = r

    if _max != i:
        arr[i], arr[_max] = arr[_max], arr[i]
        heapify(arr, n, _max)


def heap_sort(arr):
    n = len(arr)
    for i in range(n-1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr


# Quick sort, pick element as pivot, partition array into 2
# smaller than pivot and larger than pivot
# then sort 2 arrays and combine them
def partition(arr, l, h):
    idx = random.randint(l, h)
    pivot = arr[idx]
    arr[h], arr[idx] = arr[idx], arr[h]

    idx = l
    for i in range(l, h):
        if arr[i] <= pivot:
            arr[i], arr[idx] = arr[idx], arr[i]
            idx += 1

    arr[idx], arr[h] = arr[h], arr[idx]
    return idx


def quick_sort(arr, l, h):
    if l < h:
        idx = partition(arr, l, h)
        quick_sort(arr, l, idx - 1)
        quick_sort(arr, idx + 1, h)
        return arr


# Counting sort
def counting_sort(arr):
    c = [0 for i in range(256)]
    out = [0 for i in range(256)]

    # count appearance
    for x in arr:
        c[x] += 1

    for i in range(1, 256):
        c[i] += c[i-1]

    for x in arr:
        out[c[x]] = x
        c[x] += 1

    ans = [out[i] for i in range(len(arr))]
    return ans


arr = [3, 1, 16, 2, 10, 5]
# print "original array...", arr
# print "selection sort...", selection_sort(arr[:])
# print "bubble sort....", bubble_sort(arr[:])
print "insertion sort...", insertion_sort(arr[:])
# print "merge sort....", merge_sort(arr[:], 0, len(arr)-1)
# print "heap sort....", heap_sort(arr[:])
# print "quick sort....", quick_sort(arr[:], 0, len(arr)-1)
# print "counting sort....", counting_sort(arr)
