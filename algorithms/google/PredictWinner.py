

# Predict winner, each player choose from 2 end of array, predict if first
# player can win
# https://leetcode.com/problems/predict-the-winner/submissions/
# - Use DFS: if player 1 he can choose either left or right, for player 2
# whatever pick should make first player win use "and" vs "or" operation
# - Solution 2: for range (s, e), calculate how more point player 1 than play 2
# max nums[s] - winner(s+1, e), nums[e] - winner(s, e - 1), use dictionary to
# memorize => run time 0(n^2)
# - Solution 3: 
# use dynamic programing: dp[i][j] = max(nums[i] - dp[i+1][j],
# nums[j] - dp[i][j-1]) return dp[0][n-1] >= 0
# Optimize: start filling from bottom right up, dp[4] = nums[4],
# dp[3] = nums[3], then dp[4] = max(nums[4] - dp[3], nums[3] - dp[4]), so on...
class PredictSolution:
    def PredictTheWinner(self, nums):
        if not nums:
            return True

        n = len(nums)

        def dfs(l, r, s1, s2, idx):
            if idx == n:
                return s1 >= s2

            if idx % 2 == 1:
                return (dfs(l + 1, r, s1, s2 + nums[l], idx + 1) and
                        dfs(l, r - 1, s1, s2 + nums[r], idx + 1))
            else:
                return (dfs(l + 1, r, s1 + nums[l], s2, idx + 1) or
                        dfs(l, r - 1, s1 + nums[r], s2, idx + 1))

        return dfs(0, n-1, 0, 0, 0)

    def winner2(self, nums):
        n = len(nums)
        dp = [[0]*n for i in range(n)]

        for s in range(n - 1, -1, -1):
            for e in range(s+1, n):
                dp[s][e] = max(nums[s] - dp[s+1][e], nums[e] - dp[s][e-1])
        return dp[0][n-1] >= 0


    def winner3(self, nums):
        if not nums:
            return True

        n = len(nums)
        dp = [0 for _ in range(n)]

        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                dp[j] = max(nums[i] - dp[j], nums[j] - dp[j-1])
        return dp[n-1] >= 0
