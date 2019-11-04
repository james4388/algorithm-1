from collections import defaultdict


# https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
# Remove stone if they are same row or column
# Solution: build graph, for each cluster of size n we can remove n - 1 stone
# => find number of connected components by DFS
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)
        ans = n
        graph = defaultdict(list)

        for i in range(n):
            for j in range(i+1, n):
                if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                    graph[i].append(j)
                    graph[j].append(i)

        visited = set()
        count = 0

        def dfs(start):
            visited.add(start)

            for neighbor in graph[start]:
                if neighbor not in visited:
                    dfs(neighbor)

        for idx in range(n):
            if idx not in visited:
                count += 1
                dfs(idx)
        return ans - count
