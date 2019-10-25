from collections import Counter
import heapq


# https://leetcode.com/problems/top-k-frequent-elements/
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        return [x[0] for x in Counter(nums).most_common(k)]

    def topKFrequent2(self, nums, k):
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
