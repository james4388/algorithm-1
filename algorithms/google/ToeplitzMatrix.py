

# https://leetcode.com/problems/toeplitz-matrix/
# All diagonal from top left to bottom right has same value
# diagonal has same row - col
# Follow up:
# 1. What if the matrix is stored on disk, and the memory is limited such that you can only load at most one row of the matrix into the memory at once?
# Compare half of 1 row with half of the next/previous row.
# 2. What if the matrix is so large that you can only load up a partial row into the memory at once?
# Hash 2 rows (so only 1 element needs to be loaded at a time) and compare the results, excluding the appropriate beginning or ending element.
class Solution:
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        if not matrix:
            return True

        hmap = {}
        for r, row in enumerate(matrix):
            for c, val in enumerate(row):
                if r - c not in hmap:
                    hmap[r-c] = val
                elif hmap[r-c] != val:
                    return False
        return True
