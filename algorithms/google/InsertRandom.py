from random import randint


# https://leetcode.com/problems/insert-delete-getrandom-o1/
# - Implement data structure that allow insert delete and random in constant time
# Solution: use hashmap store value to index and array to store value
# - delete: swap last element to current element if current is not last
# - random: random from 0 to len - 1
class RandomizedSet:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = []
        self.cache = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.cache:
            return False

        self.arr.append(val)
        self.cache[val] = len(self.arr) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.cache:
            return False

        idx = self.cache.pop(val)
        if idx != len(self.arr) - 1:
            last = self.arr[-1]
            self.arr[idx] = last
            self.cache[last] = idx

        self.arr.pop()
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        idx = randint(0, len(self.arr) - 1)
        return self.arr[idx]
