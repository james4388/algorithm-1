from operator import truediv, mul, add, sub


# https://leetcode.com/problems/24-game/
# 4 numbers and 4 operation + - * /, check if it can produce 24
# Solution: choose 2 from 4 number, calculate, add result and recursively
# Runtime: choose 2 from 4 => 12, 2 from 3 => 6, 2 from 2 => 2
# 12 * 6 * 2 * 4 * 4 * 4 
class Solution(object):
    def judgePoint24(self, A):
        if not A: return False
        if len(A) == 1: return abs(A[0] - 24) < 1e-6

        for i in xrange(len(A)):
            for j in xrange(len(A)):
                if i != j:
                    B = [A[k] for k in xrange(len(A)) if i != k != j]
                    for op in (truediv, mul, add, sub):
                        if (op is add or op is mul) and j > i: continue
                        if op is not truediv or A[j]:
                            B.append(op(A[i], A[j]))
                            if self.judgePoint24(B): return True
                            B.pop()
        return False
