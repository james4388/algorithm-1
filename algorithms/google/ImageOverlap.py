

# https://leetcode.com/problems/image-overlap/
# 2 images represent by 2 square matrix, move one image up, down, right, left
# by any step, return maximum overlap number of bit 1
# Solution: use hashset to store index of bit 1, translate one image and count
# number of bit one overlap in other image
def largestOverlap(A, B):
    row, col = len(A), len(A[0])
    abits = set()
    bbits = set()

    for x in range(row):
        for y in range(col):
            if A[x][y]:
                abits.add((x, y))
            if B[x][y]:
                bbits.add((x, y))

    ans = 0
    for dx in range(-row + 1, row):
        for dy in range(-col + 1, col):
            count = 0
            for (i, j) in abits:
                if (i + dx, j + dy) in bbits:
                    count += 1
            ans = max(ans, count)
    return ans
