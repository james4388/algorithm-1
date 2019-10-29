import itertools


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
# Area(B, C, D) = B + C + D - BC - CD - BD + BCD
def rectangleArea(self, rectangles):
    def intersect(rec1, rec2):
        return [max(rec1[0], rec2[0]),
                max(rec1[1], rec2[1]),
                min(rec1[2], rec2[2]),
                min(rec1[3], rec2[3])]

    def area(rec):
        dx = max(0, rec[2] - rec[0])
        dy = max(0, rec[3] - rec[1])
        return dx * dy

    ans = 0
    for size in xrange(1, len(rectangles) + 1):
        for group in itertools.combinations(rectangles, size):
            ans += (-1) ** (size + 1) * area(reduce(intersect, group))

    return ans % (10**9 + 7)
