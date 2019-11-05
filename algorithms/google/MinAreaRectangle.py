

# https://leetcode.com/problems/minimum-area-rectangle/
# Give set of points on 2D plane, find minimal rectangle
# Solution: consider 2 points as diagonal of rectangle, check if other two points
# in rectangle, calculate area
def minAreaRect(points):
    s = set(map(tuple, points))

    n = len(points)
    ans = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            if p1[0] != p2[0] and p1[1] != p2[1] and (p1[0], p2[1]) in s and
                (p2[0], p1[1]) in s:
                ans = min(ans, (p2[0] - p1[0]) * (p2[1] - p1[1]))
    return ans if ans < float('inf') else -1
