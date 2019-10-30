

# https://leetcode.com/problems/course-schedule/submissions/
# Find if can finish course with prerequisites (1, 2), (2, 3)
# Solution: DFS (topological sort) check if there circle return empty
# visited: 0 not visited, 1 visited, -1 being visiting, when encounter -1 return false
def canFinish(numCourses, prerequisites):
    neighbors = [[] for i in range(numCourses)]
    visited = [0 for i in range(numCourses)]
    order = []

    for x, y in prerequisites:
        neighbors[x].append(y)

    def dfs(start):
        if visited[start] == -1:
            return False

        if visited[start] == 1:
            return True
        # Being visiting
        visited[start] = -1
        for neighbor in neighbors[start]:
            if not dfs(neighbor):
                return False
        visited[start] = 1
        order.append(start)
        return True

    for course in range(numCourses):
        if not dfs[course]:
            return []
    return order[::-1]
