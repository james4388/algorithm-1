

# Maximum vactions days
# https://leetcode.com/problems/maximum-vacation-days
# n destination and k weeks, flights: matrix nxn for n flight connection
# days: nxk maximum days to stay at city i week j
# dp[i][j] is maximum vacations at city i, week j
# dp[i][j] = max(dp[i][j], days[i][j] + dp[x][j-1]) if flights[x][i] or x == i
# base case: j = 0, dp[i][0] = days[i][0] if flights[0][i]
# optimize: j depends on j - 1, just need an array dp[j-1] and dp[j] to store
# previous and current result
def maxVacationDays(flights, days):
    n, k = len(days), len(days[0])

    dp = [[0]*k for _ in range(n)]

    dp[0][0] = days[0][0]
    for i in range(1, n):
        if flights[0][i]:
            dp[i][0] = days[i][0]

    for i in range(n):
        for j in range(1, k):
            for x in range(n):
                if flights[x][i] or x == i:
                    dp[i][j] = max(dp[i][j], days[i][j] + dp[x][j-1])
    ans = 0
    for i in range(n):
        ans = max(ans, dp[i][k-1])

    return ans
