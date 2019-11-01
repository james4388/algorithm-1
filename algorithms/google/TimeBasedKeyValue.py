# https://leetcode.com/problems/time-based-key-value-store/
from bisect import bisect
from collections import defaultdict


class TimeMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.d = defaultdict(list)


    def set(self, key, value, timestamp):
        """
        :type key: str
        :type value: str
        :type timestamp: int
        :rtype: None
        """
        self.d[key].append((timestamp, value))


    def get(self, key, timestamp):
        """
        :type key: str
        :type timestamp: int
        :rtype: str
        """
        if key not in self.d:
            return ""
        idx = bisect(self.d[key], (timestamp, chr(127)))
        return self.d[key][idx - 1][1] if idx else ""
