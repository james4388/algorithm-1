from collections import defaultdict


# HARD: Bus route
# https://leetcode.com/problems/bus-routes/
def numBusesToDestination(routes, S, T):
    n = len(routes)
    start = set()
    end = set()
    buses = [set(route) for route in routes]
    conns = defaultdict(set)

    for idx, bus in enumerate(buses):
        if S in bus:
            start.add(idx)
        if T in bus:
            end.add(idx)

    for i in range(1, n):
        for j in range(i):
            if buses[i] & buses[j]:
                conns[i].add(j)
                conns[j].add(i)

    def bfs(node):
        q = deque([(node, 1)])
        visit = set([node])
        while q:
            bus, num = q.popleft()
            if bus in end:
                return num

            for conn in conns.get(bus, []):
                if conn not in visit:
                    q.append((conn, num + 1))
                    visit.add(conn)
        return -1

    ans = float('inf')
    for node in start:
        num = bfs(node)
        if num != -1:
            ans = min(ans, num)
    return ans if ans < float('inf') else -1

print("num of buses...", numBusesToDestination([[1, 2, 7], [3, 6, 7]], 1, 6))
