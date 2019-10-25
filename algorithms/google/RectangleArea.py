

# Rectangle area
# https://leetcode.com/problems/rectangle-area/submissions/
# Find overlap area, left = max(A, E) = E
# for right, if overlap right = min(C, G) if non overlap, this is wrong
# min(C, G) = C < left => take max(min(C, G), left) = E
# Same for bottom and top
def computeArea(A, B, C, D, E, F, G, H):
    left = max(A, E)
    right = max(min(C, G), left)
    bottom = max(B, F)
    top = max(min(D, H), bottom)
    return (C-A)*(D-B) + (G-E)*(H-F) - (top - bottom) * (right - left)


# Rectangle area 2
# https://leetcode.com/problems/rectangle-area-ii/solution/
# - Solution 1: sorted x and y list, remap x, y to its index
# use 2D array to fill in cell covered by rectangle, grid[x][y] = 1 for x in
# map[x1], map[x2] and y in mapy[y1], mapy[y2]
# if grid = 1, calculate area, => run time: O(n^3)
# - Solution 2: line sweep
# Consider every rec as 2 layers, (x1, x2) open at y1 and (x1, x2) close at y2
# sort every open and close layers by y, calculate area for each layer y
# (x1, x2), (x3, x4)...xk => area = sum(x different) * (y - previous y)
# Optimize: use segment tree to add or remove layers => nlog(n)
