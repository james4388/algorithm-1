

# Find redundant connection to make tree
# https://leetcode.com/problems/redundant-connection/
# https://leetcode.com/problems/redundant-connection-ii/
# 2 cases: vertex with 2 parents, cycle in tree
def findRedundantConnection(edges):
    """
    :type edges: List[List[int]]
    :rtype: List[int]
    """
    def find(parent, x):
        if x not in parent:
            return x

        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]

    parent = {}
    p, q = None, None

    for edge in edges:
        if not parent.get(edge[1]):
            parent[edge[1]] = edge[0]
        else:
            p = parent[edge[1]], edge[1]
            q = [edge[0], edge[1]]
            edge[1] = 0

    parent = {}

    for u, v in edges:
        if v == 0:
            continue

        ur, vr = find(parent, u), find(parent, v)
        if ur == vr:
            if not p:
                return [u, v]
            return p
        parent[vr] = ur
    return q

print("redundant connection...", findRedundantConnection([[5,2],[5,1],[3,1],[3,4],[3,5]]))
