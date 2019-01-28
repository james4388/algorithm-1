

# Max profit buy and sell stock one time
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return 0

    _min = prices[0]
    profit = 0

    for item in prices:
        _min = min(item, _min)
        profit = max(item - _min, profit)

    return profit


print "max profit...", maxProfit([3, 3])


# Buy and sell multiple times
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
def maxProfit2(prices):
    if not prices or len(prices) < 1:
        return 0

    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit


# Buy and sell at most k time
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
# Bruteforce, split arry into 2, find max profit in each sub array
# DP solution: dp[i][j] profit for transaction i at day j
# dp[i, j] = max(dp[i][j-1], prices[j] - prices[k] + dp[i-1][k]) {k = 0...j}
# max(dp[i][j-1], prices[j] + max(dp[i-1][k] - prices[k]))
# dp[0, j] = 0, dp[i, 0] = 0
def maxProfit3(prices, k):
    if not prices or len(prices) < 1:
        return 0
    n = len(prices)
    if k >= n/2:
        return maxProfit2(prices)

    dp = [[0 for _ in range(n)] for _ in range(k+1)]

    for i in range(1, k+1):
        local = dp[i-1][0] - prices[0]
        for j in range(1, n):
            dp[i][j] = max(dp[i][j-1], prices[j] + local)
            local = max(dp[i-1][j] - prices[j], local)
    return dp[k][n-1]


print "max profit3...", maxProfit3([3,3,5,0,0,3,1,4], 2)
