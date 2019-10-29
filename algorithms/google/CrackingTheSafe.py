

# Cracking the safe
# https://leetcode.com/problems/cracking-the-safe/
# Give password of size n, contain digit 0...k-1
# Find shortest string to unlock
# Solution: n = 2, k = 2, sequence: 00 -> 01 -> 11-> 10, ans = 00110
# take n - 1 char and append new number at the end, check if new sequence visited
# add it to answer
def crackSafe(n, k):
    visited = set()
    ans = []

    def cycle(node):
        for x in map(str, range(k)):
            v = node + x
            if v not in visited:
                visited.add(v)
                cycle(v[1:])
                ans.append(x)

    cycle("0" * (n-1))

    return "".join(ans) + "0"*(n-1)


def crackSafeIteractive(n, k):
    ans = "0" * (n - 1)
    visits = set()
    for x in range(k ** n):
        current = ans[-n+1:] if n > 1 else ''
        for y in range(k - 1, -1, -1):
            if current + str(y) not in visits:
                visits.add(current + str(y))
                ans += str(y)
                break
    return ans

print("cracking the safe...", crackSafeIteractive(1, 2))
