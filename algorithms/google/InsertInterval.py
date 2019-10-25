

# Insert interval, give a list of sorted intervals, insert new interval into
# list and merge if overlap
# HARD: https://leetcode.com/problems/insert-interval/description/
# Loop through list, find all overlap with new interval, and merge them
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return '{}->{}'.format(self.start, self.end)


class IntervalSolution(object):
    def isOverlap(self, x, y):
        return (x.end >= y.start) and (y.end >= x.start)

    def merge(self, x, y):
        return Interval(min(x.start, y.start), max(x.end, y.end))

    def insert(self, intervals, newInterval):
        if not intervals:
            return [newInterval]
        res = []
        i, n = 0, len(intervals)
        while i < n:
            if newInterval.end < intervals[i].start:
                break

            if self.isOverlap(newInterval, intervals[i]):
                newInterval = self.merge(newInterval, intervals[i])
            else:
                res.append(intervals[i])
            i += 1

        res.append(newInterval)
        while i < n:
            res.append(intervals[i])
            i += 1
        return res

    def printIntervals(self, intervals):
        l = [str(x) for x in intervals]
        print("intervals...", l)
