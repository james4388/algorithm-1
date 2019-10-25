import random
import bisect


# Random pick with weight
# https://leetcode.com/problems/random-pick-with-weight/
# Given array with weight, random pick an index according to its weight
# Solution: create sub array with prefix sum of weight
# use binary search for random value in prefix => runtime logn
class PickSolution:
    def __init__(self, w):
        p = w[:]
        for i in range(1, len(p)):
            p[i] += p[i-1]

        self.p = p
        self.max = p[-1]

    def pickIndex(self):
        val = random.randrange(self.max) + 1
        return bisect.bisect_left(self.p, val)
