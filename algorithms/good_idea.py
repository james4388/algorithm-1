

# Smallest range:
# https://leetcode.com/problems/smallest-range-ii
# Add K or -K to each number, find smallest possible max - min
# Observe: a < b, need to compare (a + K, max - K) and (b - K, min + K)
# If better than answer, update result
class SmallestRangeSolution:
    def smallestRangeII(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        A.sort()
        mi, ma = A[0], A[-1]
        res = ma - mi
        for i in range(len(A) - 1):
            a, b = A[i], A[i+1]
            res = min(res, max(a + K, ma - K) - min(b - K, mi + K))
        return res
