

# K empty slots
# HARD: https://leetcode.com/problems/k-empty-slots
# Given bloom array: [1, 3, 2] day 1 flower 1, day 2 flower 3, day 3 flower 2
# Given k find which day, there's two blooming and k in between not blooming
# Convert to array: days[x] = i for flower x bloom at day i
# Find interval: left, right which is minimum bloom day of this interval,
# solution 1: use sliding min queue, min of window > max(left, right)
# solution 2: sliding window, for interval left, right, if found days[i] < left
# or days[i] < right, update the window to i, i+k+1
def kEmptySlots(flowers, k):
    days = [0]*len(flowers)

    for day, flower in enumerate(flowers, 1):
        days[flower - 1] = day

    left, right = 0, k + 1
    res = len(flowers) + 1
    while right < len(days):
        for i in range(left, right+1):
            if days[i] < days[left] or days[i] < days[right]:
                left, right = i, i + k + 1
                break
        else:
            res = min(res, max(left, right))
            left, right = right, right + k + 1
    return res if res < len(flowers) + 1 else -1
