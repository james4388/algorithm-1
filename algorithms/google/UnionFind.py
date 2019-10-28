# "Find" on parent dictionary, if parent[v] = v return v
# else parent[v] = find(parent[v])
# "Union" edge: u -> v, u father of v, find parent of u = parent_u,
# parent[v] = parent_of_u
# Path compression and union by rank
class UnionFind:
    # Compress path as finding
    def __init__(self, num):
        self.parent = list(range(num))
        self.rank = [0 for i in range(num)]
        # Size of each cluster
        self.size = [1 for i in range(num)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # Optimize rank on union, choose parent node with higher rank
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)

        # Cannot union, there's cycle
        if xr == yr:
            return False

        if self.rank[yr] < self.rank[xr]:
            xr, yr = yr, xr

        if self.rank[xr] == self.rank[yr]:
            self.rank[xr] += 1

        self.parent[yr] = xr
        self.size[xr] += self.size[yr]
        return True


def detectCycle(edges):
    uf = UnionFind(len(edges))
    for edge in edges:
        if not uf.union(*edge):
            return edge
    return False
