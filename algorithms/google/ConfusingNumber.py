# https://leetcode.com/problems/confusing-number-ii/
# Number: 0, 1, 6, 8, 9 can rotate to become different number, count how many
# Solution: For adding new number, we can calculate its rotation by putting
# its rotate at begin and add previous rotations
# For example:
# 1, 1 -> 1*10 + 6, -> 9 * 10 + 1 = 91
# 16, 91 + 1 -> 16*10 + 1 = 161, 1* 10 ^ 2 + 91 = 191
def confusingNumber(N):
    count = [0]
    pairs = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}

    def dfs(num, rotation, length):
        if num > N:
            return
        if num != rotation:
            count[0] += 1
        for x in pairs:
            dfs(num * 10 + x, pairs[x] * (10 ** length) + rotation, length + 1)
    dfs(1, 1, 1)
    dfs(6, 9, 1)
    dfs(8, 8, 1)
    dfs(9, 6, 1)
    return count[0]
