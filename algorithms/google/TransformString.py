

# Number of ways to remove 0 or more characters to transform one
# string to another string
# e.g: abcccdf -> abccdf = 3, aabba -> ab = 4
# Call dp[m][n] = result
# sequence x1..xi, y1...yj
# dp[0][j] = 1, deleting all characters in y
# xi != yj, dp[i][j] = dp[i][j-1] match sequence y1...y(j-1)
# xi == yj, can match either sequence y1...y(j-1) to x1...x(i-1)
# or match y1...y(j-1) to x1...xi
def numWayTransform(x, y):
    if not x:
        return 1

    m, n = len(x), len(y)
    dp = [[0 for i in range(n)] for j in range(m)]

    for i in range(0, m):
        for j in range(i, n):
            if i == 0:
                if x[i] == y[j] and j == 0:
                    dp[i][j] = 1
                elif x[i] == y[j]:
                    dp[i][j] = dp[i][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
            else:
                if x[i] != y[j]:
                    dp[i][j] = dp[i][j-1]
                else:
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]

    return dp[m-1][n-1]
