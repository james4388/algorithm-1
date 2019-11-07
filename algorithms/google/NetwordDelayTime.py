from collections import defaultdict
from heapq import heappop, heappush

# https://leetcode.com/problems/network-delay-time/
# Give network of directed edge, find time to transfer to all node, from start node
# Solution: use heap to store distance and current node, each time pop a node
# check if it's not visited, increase count, if count = length, return current timestamp
# add its neighbors into heap and plus timestamp
class Solution:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        network = defaultdict(list)
        for (src, dest, t) in times:
            network[src].append((dest, t))
        count = 0
        visited = [False for i in range(N+1)]
        queue = [(0, K)]
        while queue:
            ts, node = heappop(queue)
            if visited[node]:
                continue

            visited[node] = True
            count += 1
            if count == N:
                return ts

            for neighbor, t in network[node]:
                heappush(queue, (ts + t, neighbor))
        return -1
