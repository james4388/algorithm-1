from collections import deque


# Flatten iterator of iterators
# Use array to remove ended iterator -> not efficient, use queue to enqueue and
# dequeue
class IF:
    def __init__(self, iterators):
        self.queue = deque(iterators)

    def __iter__(self):
        return self

    def next(self):
        if not self.queue:
            raise StopIteration

        while True:
            try:
                it = self.queue.popleft()
                item = next(it)
                self.queue.append(it)
                return item
            except StopIteration:
                continue
            except IndexError:
                raise StopIteration
            else:
                break

# iterators = [iter(range(10)), iter(range(10, 20, 2)), iter(range(30, 50, 3))]
# it = IF(iterators)
# for i in range(25):
#     print("next...", next(it))
