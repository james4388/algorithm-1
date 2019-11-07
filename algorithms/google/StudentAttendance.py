

# https://leetcode.com/problems/student-attendance-record-ii/
# Student record: A absent, P present, L late
# Rewardable: no more than 2 A and no more than 2 continuous L
# Give length N: find how many string for student records

# Solution: to create valid records, we can put A in position 0...n-1 or no A
# and ans = leftA * rightA for A from 0 ... n-1, and left, right part contains only P and L
# count how many records that contains P, L for no more than 2 L
# call dp[k] is count at length k, 3 cases:
# end with P always rewardable => check dp[k-1]
# end with L: 2 case PL => always rewardable => dp[k-2]
#  LL: check if dp[k-3]
# dp[k] = dp[k-1] + dp[k-2] + dp[k-3] = 2*dp[k-1] - dp[k-4]
class Solution:

    def checkRecord(self, n: int) -> int:
        M = 1000000007
        size = max(6, n + 1)
        f = [0 for i in range(size)]
        f[0] = 1
        f[1] = 2
        f[2] = 4
        f[3] = 7

        for i in range(4, n+1):
            f[i] = (2 * f[i-1]) % M + ( M - f[i-4]) % M

        ans = f[n]
        for i in range(1, n + 1):
            ans += (f[i-1] * f[n-i]) % M
        return ans % M
