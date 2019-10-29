from collections import defaultdict



# https://leetcode.com/problems/number-of-boomerangs/
# Find i, j , k for distance i, j = distance i, k
# Solution: use hash map to store distance, num = count * (count - 1)
class Solution:
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        def distance(x, y):
            return (y[0] - x[0])**2 + (y[1] - x[1])**2

        n = len(points)
        ans = 0

        for i in range(n):
            table = defaultdict(int)
            for j in range(n):
                d = distance(points[j], points[i])
                table[d] += 1

            for k in table:
                ans += table[k] * (table[k] - 1)
        return ans
