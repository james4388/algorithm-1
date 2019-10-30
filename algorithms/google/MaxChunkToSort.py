

# https://leetcode.com/problems/max-chunks-to-make-sorted/
# Give array size n, each value from 0,...,n-1, split array into chunks and sort them
# then concat chunks to have sorted array, find max chunk can make
# Solution: check if max value = index, increase count
class Solution(object):
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        ans = ma = 0
        for i, x in enumerate(arr):
            ma = max(ma, x)
            if ma == i: ans += 1
        return ans

    # https://leetcode.com/problems/max-chunks-to-make-sorted-ii/
    # Array value can be duplicate
    # Solution: observe that we can split at one point if maximum value in that
    # left sub array <= minimum value of the right subarray
    # use subarray to pre-calculate min from right
    # if max at idx <= min at idx + 1, or we go to end of array => ans += 1
    def maxChunksToSortedII(self, arr):
        n = len(arr)
        right = [0 for i in range(n)]
        _min = arr[-1]
        for i in range(n-1, -1, -1):
            _min = min(arr[i], _min)
            right[i] = _min

        _max = arr[0]
        ans = 0
        for idx, val in enumerate(arr):
            _max = max(_max, val)
            if idx == n - 1 or _max <= right[idx+1]:
                ans += 1
        return ans
