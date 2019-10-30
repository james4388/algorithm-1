from bisect import bisect_left as bl, bisect_right as br


# https://leetcode.com/problems/range-module/
# Design module to add, remove and query a range
# Solution: store all start and end in one sorted array, start at even index,
# end at odd index
# Use binary search for each operations:
# - Add: add start, end if it's out of end, index is even
# - Remove: keep left, right if they are in beetween of interval, index odd
# - Query: return true if start and end in one interval, use bisect right, left
# idx left = idx right and odd
class RangeModule:

    def __init__(self):
        self._X = []

    def addRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i%2 == 0) + [right]*(j%2 == 0)

    def queryRange(self, left, right):
        i, j = br(self._X, left), bl(self._X, right)
        return i == j and i%2 == 1

    def removeRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i%2 == 1) + [right]*(j%2 == 1)
