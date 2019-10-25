import random


# Read 4 char
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
        self.buffStr = []
        self.readStr = 0

    def copy_from(self, buff):
        idx = 0
        length = read4(buff)
        while idx < length:
            self.buffStr.append(buff[idx])
            idx += 1
        self.readStr += length
        return length

    def read(self, buff, n):
        tmp = range(4)
        length = self.copy_from(tmp)

        need_read = 0
        if self.readStr < n:
            need_read = n - self.readStr

        while need_read > 0 and length > 0:
            length = self.copy_from(tmp)
            need_read -= length

        idx = 0
        while idx < min(self.readStr, n):
            buff[idx] = self.buffStr[idx]
            idx += 1
        return idx

r4 = Read4Solution()
# buff = list(range(1028))
# r4.read(buff, 10)
#
# r4.read(buff, 20)
#
# r4.read(buff, 25)
#
