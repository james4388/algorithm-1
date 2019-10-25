

# Cracking the safe
# https://leetcode.com/problems/cracking-the-safe/
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

print("cracking the safe...", crackSafe(1, 2))
