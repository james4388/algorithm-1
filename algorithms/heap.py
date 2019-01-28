from collections import deque


# Heap data structure implementation
class Heap(object):
    def __init__(self, data=None):
        if not data:
            self.data = []
            self.len = 0
        else:
            self.data = data
            self.len = len(data)
            self.build_heap(data)

    def heapify(self, fro):
        left = 2*fro + 1
        right = 2*fro + 2
        max_ = fro
        if left < self.len and self.data[max_] < self.data[left]:
            max_ = left
        if right < self.len and self.data[max_] < self.data[right]:
            max_ = right
        if max_ != fro:
            self.data[fro], self.data[max_] = self.data[max_], self.data[fro]
            self.heapify(max_)
        return

    def insert(self, val):
        self.data.append(val)
        self.len += 1
        i = self.len/2
        while i > 0:
            self.heapify(i)
            i = i/2

    @staticmethod
    def sort(data):
        h = Heap(data)
        l = deque([])
        for i in range(h.len-1, -1, -1):
            if h.len == 1:
                l.appendleft(h.data.pop())
                break
            h.data[i], h.data[0] = h.data[0], h.data[i]
            l.appendleft(h.data.pop())
            h.len -= 1
            h.heapify(0)
        return l

    def build_heap(self, l):
        for i in range(self.len/2, -1, -1):
            self.heapify(i)

    def __str__(self):
        return "{0}".format(self.data)


# Check k largest equal or smaller than number x
def compare_heap(heap, idx, k, x):
    print "Compare at...", idx, k
    if k <= 0 or idx < 0 or idx > heap.len:
        return k

    if heap.data[idx] > x:
        k = compare_heap(heap, 2*idx + 1, k-1, x)
        k = compare_heap(heap, 2*idx + 2, k, x)
    return k


arr = [0, 9, 1, 2, 4, 8, 7, 3, 99, -1]
h = Heap(arr)
print "Heap...", h
print "Heap sort...", Heap.sort([0, 9, 1, 2, 4, 8, 7, 3, 99, -1])
print "check 4 smallest larger than 5...", compare_heap(h, 0, 4, 5)
