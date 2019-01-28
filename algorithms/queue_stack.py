from collections import deque


class Queue(deque):
    def enqueue(self, item):
        self.append(item)

    def dequeue(self):
        return self.popleft()

    def is_empty(self):
        return self.__len__() == 0


class Stack(list):
    def push(self, item):
        self.append(item)

    def is_empty(self):
        return self.__len__() == 0
