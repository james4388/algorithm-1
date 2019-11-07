
# Minimize largest continuous sum array
# https://leetcode.com/problems/split-array-largest-sum/
# Call dp[i][j] is largest sum at index i, split into j array
# dp[i][j] = min( max(dp[x][j-1], sum x...i for x in range j - 1, i - 1)
# Base case: dp[i][1] = sum 0...i
# dp[1][j] = arr[0]
# Optimize: use binary search
# min value is max(arr), max value is sum(arr)
# use bst for value of mid = (lo + high)/2 and check if can make more than
# m cut then set high = mid, if cannot set lo = mid + 1
# Run time: check if can cut O(n), binary search O(log(sum(arr)))
class SplitArraySolution:
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        n = len(nums)
        dp = [[sys.maxint for i in range(m+1)] for j in range(n+1)]
        dp[0][0] = 0
        acc = [0 for i in range(n+1)]
        for i in range(1, n+1):
            acc[i] = acc[i-1] + nums[i-1]

        for i in range(1, n+1):
            for j in range(1, m+1):
                for x in range(i):
                    dp[i][j] = min(dp[i][j], max(dp[x][j-1], acc[i] - acc[x]))
        return dp[n][m]

    # Binary search solution
    def splitArray2(self, nums, m):
        def valid(mid):
            cnt = 0
            current = 0
            for n in nums:
                current += n
                if current > mid:
                    cnt += 1
                    if cnt >= m:
                        return False
                    current = n
            return True

        l = max(nums)
        h = sum(nums)

        while l < h:
            mid = l+(h-l)/2
            if valid(mid):
                h = mid
            else:
                l = mid+1
        return l

s = SplitArraySolution()
print "split array...", s.splitArray([7, 2, 5, 10, 8], 4)
