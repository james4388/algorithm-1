import random


# HARD: Read 4 char
# https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times
# Everytime call read4(buff) return number of char read
# Write function read n characters using read4, call multiple times
# Use buffStr to store read string, if read string less than n, read more
# string into buffer until no more string
def read4(buff):
    strs = 'abcdefghijkmnopqrtsxyz'
    for i in range(4):
        idx = random.randint(0, len(strs) - 1)
        buff[i] = strs[idx]
    return i


class Read4Solution(object):
    def __init__(self):
        self.cache = deque()

    def copy(self):
        buff = range(4)
        read = read4(buff)
        for i in range(read):
            self.cache.append(buff[i])
        return read

    def read(self, buff, n):
        read = -1

        while len(self.cache) < n and read != 0:
            read = self.copy()

        total = min(len(self.cache), n)
        for i in range(total):
            buff[i] = self.cache.pop(left)
        return total

r4 = Read4Solution()
# buff = list(range(1028))
# r4.read(buff, 10)
#
# r4.read(buff, 20)
#
# r4.read(buff, 25)
#
