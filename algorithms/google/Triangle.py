

# Find minimum sum path in triangle
# https://leetcode.com/problems/triangle/description/
# Process level by level, for each element it can add previous row with same
# column j or j - 1, except at index 0 and n-1
def triangle(numlist):
    if not numlist:
        return -1

    curr = numlist[0]
    n = len(numlist)
    for i in range(1, n):
        tmp = numlist[i]
        for j in range(len(tmp)):
            if j == 0:
                tmp[j] += curr[j]
            elif j == len(tmp) - 1:
                tmp[j] += curr[j-1]
            else:
                tmp[j] += min(curr[j], curr[j-1])
        curr = tmp
    return min(curr)

print "triangle....", triangle([[2], [3, 4], [6, 5, 7], [4, 1, 3, 8]])
