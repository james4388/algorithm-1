

# https://leetcode.com/problems/optimize-water-distribution-in-a-village/
# Distribute water to village, by pipes or wells
# Solution: preset well as a node with edge to other house, weight is cost to build
# well, sorted by cost, process each, use union find, if x != y, union them and add cost
 def minCostToSupplyWater(self, n, wells, pipes):
    uf = {i: i for i in xrange(n + 1)}

    def find(x):
        if x != uf[x]:
            uf[x] = find(uf[x])
        return uf[x]

    w = [[c, 0, i] for i, c in enumerate(wells, 1)]
    p = [[c, i, j] for i, j, c in pipes]
    res = 0
    for c, x, y in sorted(w + p):
        x, y = find(x), find(y)
        if x != y:
            uf[find(x)] = find(y)
            res += c
            n -= 1
        if n == 0:
            return res
