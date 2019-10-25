

# Special keyboard: key 1 => 'A', key 2 => Ctrl + A, key 3 => Ctrl + C
# key 4: Ctrl + V, can do multiple copy key4
# C(N) is result, C(0) = 0, C(1) = 1, C(i) = max(C(i-1) + 1, C(i-3) * 2)
# Consider user can copy multiple times, use `copy` array to cache the string
# if at happen at index i-1, then can copy again: val = dp[i-1] + copy[i-1]
# copy[i] = copy[i-1]
# if not then, have to issue 3 key: copy[i] = copy[i-3], val = dp[i-3] * 2
def specialKeyboard(N):
    if N <= 0:
        return 0

    dp = [0 for _ in range(N+1)]
    dp[1] = 1
    copy = 1
    for i in range(2, N+1):
        if i < 3:
            dp[i] = dp[i-1] + 1
        else:
            x = dp[i-1] + 1
            y = dp[i-1] + copy
            z = dp[i-3] * 2
            dp[i] = max(x, y, z)
            if dp[i] == z:
                copy = dp[i-3]

    return dp[N]
