from collections import deque


# Give 2 list [1, 2], [3, 4, 5], alternatively return each element when call next()
# Use a queue, put (nth array, index) into queue for each array
# Call next: pop one element and increase its index, push back into queue if index < length
class ZigzagIterator:
    def __init__(self, arrs):
        self.arrs = arrs
        self.queue = deque()

        for i in range(len(arrs)):
            if not arrs[i]:
                continue
            self.queue.append((i, 0))

    def next(self):
        if not self.queue:
            raise StopIteration

        nth, idx = self.queue.popleft()
        val = self.arrs[nth][idx]
        if idx < len(self.arrs[nth]):
            self.queue.append((nth, idx + 1))
        return val

    def hasNext(self):
        return len(self.queue) > 0
