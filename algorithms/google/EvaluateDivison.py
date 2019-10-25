from collections import defaultdict


# Division evaluate
# https://leetcode.com/problems/evaluate-division/description/
# Use DFS with hash table storing division pairs
class DivisionSolution:
    def dfs(self, start, end, lookup, status, res):
        if end in lookup[start]:
            return res * lookup[start][end]
        for k in lookup[start]:
            if k not in status:
                r = self.dfs(k, end, lookup, status + [k], res*lookup[start][k])
                if r is not None:
                    return r
        return None

    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        lookup = defaultdict(dict)
        for formular in zip(equations, values):
            a, b = formular[0]
            res = formular[1]
            lookup[a][b] = res
            if res != 0:
                lookup[b][a] = 1.0/res
        out = []

        for query in queries:
            x, y = query
            if x not in lookup or y not in lookup:
                out.append(-1.0)
                continue

            if x == y:
                out.append(1.0)
                continue

            if y in lookup[x]:
                out.append(lookup[x][y])
                continue

            res = self.dfs(x, y, lookup, [], 1.0)
            out.append(res or -1.0)
        return out
