from heapq import heappush, heappop


# https://leetcode.com/problems/campus-bikes/
# Give list of worker and bike on campus, assign worker to a bike for shortest
# mahattan distance d = |x1 - x2| + |y1 - y2|
# Solution: calculate distance from each worker to each bike, put into heap
# pop from heap check if both worker and bike are not assigned
# Optimize: if value of x, y < 1000, can use count sort, use distance array size 2000
# push pair of (worker, bike) into array, loop through array and assign if available
def assignBikes(workers, bikes):
    assignedWorkers = [-1 for i in range(len(workers))]
    assignedBikes = [-1 for i in range(len(bikes))]
    queue = []

    for i, worker in enumerate(workers):
        for j, bike in enumerate(bikes):
            dis = abs(worker[0] - bike[0]) + abs(worker[1] - bike[1])
            heappush(queue, (dis, i, j))

    assigned = 0
    while assigned < len(workers):
        (_, w, b) = heappop(queue)
        if assignedWorkers[w] == -1 and assignedBikes[b] == -1:
            assignedWorkers[w] = b
            assignedBikes[b] = w
            assigned += 1
    return assignedWorkers

if __name__ == '__main__':
    ans = assignBikes([[0,0],[1,1],[2,0]], [[1,0],[2,2],[2,1]])
    print("assigned...", ans)
