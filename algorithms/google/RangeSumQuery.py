

# 1D array
# https://leetcode.com/problems/range-sum-query-immutable/
class NumArray(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        n = len(nums)
        self.sum = [0 for i in range(n+1)]
        for i in range(1, n+1):
            self.sum[i] = nums[i-1] + self.sum[i-1]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sum[j+1] - self.sum[i]


# Query range 2D array
# HARD: https://leetcode.com/problems/range-sum-query-2d-immutable/description/
# Store sum from 0,0 to i, j by sum[i][j] = sum[i-1][j] + sum[i][j-1] -
# sum[i-1][j-1] + matrix[i][j]
# Query: r1, c1 to r2, c2 = sum[r2][c2] - sum[r2][c1] -
# sum[r1][c2] + sum[r1][c1]
# Follow up: support update(r, c, value), calculate difference and add up to
# sub array with index equal or larger than current position
# Second solution: do prefix for each row, calculate sum by sum prefix for each
# row, update can do by re-calculate prefix for that row only
class NumMatrix(object):
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        prefix = [[0]*(n+1) for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                prefix[i][j] = (prefix[i-1][j] + prefix[i][j-1] -
                                prefix[i-1][j-1] + matrix[i-1][j-1])
        self.prefix = prefix
        self.nums = matrix
        self.row = m
        self.col = n

    def sumRegion(self, r1, c1, r2, c2):
        if (not 0 <= r1 < self.row or not 0 <= r2 < self.row
            or not 0 <= c1 < self.col or not 0 <= c2 < self.col):
            return None
        prefix = self.prefix
        return (prefix[r2+1][c2+1] - prefix[r2+1][c1] -
                prefix[r1][c2+1] + prefix[r1][c1])

    def update(self, r, c, value):
        prefix = self.prefix
        nums = self.nums
        delta = value - nums[r][c]
        nums[r][c] = value
        for i in range(r+1, self.row+1):
            for j in range(c+1, self.col+1):
                prefix[i][j] += delta


arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
nm = NumMatrix(arr)
print "query by region sum...\n", nm.sumRegion(1, 1, 2, 2)
print "update matrix...\n", nm.update(1, 1, 10)
print "query by region sum after...\n", nm.sumRegion(1, 1, 2, 2)
