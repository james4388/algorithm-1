from heap import heappop, heappush, heapify


# Minize max distance to gas station
# https://leetcode.com/problems/minimize-max-distance-to-gas-station
# On a horizontal number line, we have gas stations at positions
# add more K station so that the maximum distance between adjacent
# gas stations, is minimized.
# Use a max heap to store distance between station
# everytime add one more station to max distance, recalculate distance and push
# back to heap => run time: nlog(n) + klog(n), space: O(n)
# Optimize: each time adding one station in largest distance, can we add more?
# knowing that the number of added stations should make current distance less
# than next second distance, num += max(ceil(largest/second_largest), 1)
def maxGasStationDistance(stations, K):
    distances = []
    n = len(stations)
    for i in range(n-1):
        d = stations[i+1] - stations[i]
        heappush(distances, [-d, 1, d])

    for j in range(K):
        item = heappop(distances)
        prio, num, curr = item
        num += 1
        heappush(distances, [-(curr/num), num, curr])
    return -distances[0][0]

# Use binary search, find possible distance D that can use up to K station
def minmaxGasDist(stations, K):
    def possible(D):
        return sum(int((stations[i+1] - stations[i]) / D)
                   for i in xrange(len(stations) - 1)) <= K

    lo, hi = 0, 10**8
    while hi - lo > 1e-6:
        mi = (lo + hi) / 2.0
        if possible(mi):
            hi = mi
        else:
            lo = mi
    return lo

print("max distance....", maxGasStationDistance([0, 4, 7, 12, 18, 20], 9))
