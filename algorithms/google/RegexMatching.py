

# Regex matching + for single character or empty, * for any sequence
# e.g pattern: b+a*ab -> baaabab: True
# C(i, j) is result of string at index i, pattern at index j
# yj = '+' -> C(i, j) = C(i-1, j-1) for single match or
# C(i, j-1) for empty match
# yj = '*' -> C(i, j) = C(i, j-1) for empty sequence or C(i-1, j)
# for multiple match
# Edge case: string is empty, pattern not empty: '' and '+*'
# pattern is empty, string is not empty
def regexMatching(pattern, text):
    if not pattern and not text:
        return True

    m, n = len(text), len(pattern)
    dp = [[False] * (n+1) for _ in range(m+1)]

    dp[0][0] = True

    # edge case string is empty, pattern is not empty
    for j in range(1, n+1):
        if pattern[j-1] in ('*', '+'):
            dp[0][j] = dp[0][j-1]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if pattern[j-1] == '+':
                dp[i][j] = dp[i-1][j-1] or dp[i][j-1]
            elif pattern[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif text[i-1] == pattern[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = False
    return dp[m][n]
