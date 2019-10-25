

# Trap rain water
# HARD: https://leetcode.com/problems/trapping-rain-water/solution/
# Brute force: the amount of trap rain water equal to minimum of max left column
# and max right column - height of current column, go 2 passes to find max
# left and max right, => run time O(n^2)
# Use 2 max array to avoid run max left and max right multiple times
def trapRainWater(heights):
    if not heights:
        return 0
    length = len(heights)
    maxLeft = [0 for _ in range(length)]
    maxLeft[0] = heights[0]
    for i in range(1, length):
        maxLeft[i] = max(maxLeft[i-1], heights[i])

    maxRight = [0 for _ in range(length)]
    maxRight[length-1] = heights[length-1]
    for i in range(length-2, -1, -1):
        maxRight[i] = max(maxRight[i+1], heights[i])
    res = 0

    for i in range(length):
        res += min(maxLeft[i], maxRight[i]) - heights[i]

    return res

print "trap rain water...", trapRainWater([1,3,2,4,1,3,1,4,5,2,2,1,4,2,2])
