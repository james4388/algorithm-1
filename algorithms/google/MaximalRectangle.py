


# https://leetcode.com/problems/maximal-rectangle/
# Give array of 1, 0, find maximal rectangle
# Solution: calculate height of each column at each row level, find maximum rectangle
# in that row level
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        m = len(matrix)
        n = len(matrix[0])
        res = 0
        h = [0 for _ in range(n)]
        for i in range(m):
            stack = []
            for j in range(n):
                if matrix[i][j] == '1':
                    h[j] += 1
                else:
                    h[j] = 0
                if not stack or h[j] >= h[stack[-1]]:
                    stack.append(j)
                else:
                    while stack and h[j] < h[stack[-1]]:
                        idx = stack.pop()
                        l = j - stack[-1] - 1 if stack else j
                        area = h[idx] * l
                        res = max(res, area)
                    stack.append(j)
            while stack:
                idx = stack.pop()
                l = n - stack[-1] - 1 if stack else n
                area = h[idx] * l
                res = max(res, area)
        return res
