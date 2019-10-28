

# https://leetcode.com/problems/toeplitz-matrix/
# All diagonal from top left to bottom right has same value
# diagonal has same row - col
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
