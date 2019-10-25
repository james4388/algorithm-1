from heapq import heappush, heappop


# Minimum cost to hire K worker
# https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
# Brute force: every time choose worker as captain, calculate other workers as
# ratio with current work, sort the list and sum prices
# Optimize: use max heap for quality, calculate ratio price/quality
# (to maximize quality while maintain low price, maintain paid ratio) and sort
# in order, every time put a quality into max heap, if heap larger than K,
# then pop item with larger quality out (use max heap to lower the price)
class HireWorkerSolution:
    def mincostToHireWorkers(self, quality, wage, K):
        """
        :type quality: List[int]
        :type wage: List[int]
        :type K: int
        :rtype: float
        """
        if not quality or not wage or K <= 0:
            return -1

        if K == 1:
            return min(wage)

        workers = sorted([(float(w)/q, w, q) for w, q in zip(wage, quality)])
        pool = []
        sumq = 0
        ans = float('inf')

        for ratio, w, q in workers:
            heappush(pool, -q)
            sumq += q

            if len(pool) > K:
                sumq += heappop(pool)

            if len(pool) == K:
                ans = min(ans, sumq*ratio)
        return ans
