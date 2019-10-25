from bisect import bisect_right, insort
from collections import defaultdict


# https://leetcode.com/problems/my-calendar-i/
# Check if can book start, end time meeting
# Use sorted array to store booking
# Book: use binary search to find available time, check if it overlaps with
# left and right booking
# Other solution: use TreeMap, compare new booking with left and right
# Python: use binary tree to insert
class Book(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, other):
        return self.start < other.start

    def __eq__(self, other):
        return self.start == other.start

    def __gt__(self, other):
        return self.start > other.start


class MyCalendar(object):
    def __init__(self):
        self.booking = []

    def book(self, start, end):
        bk = Book(start, end)
        if not self.booking:
            self.booking.append(bk)
            return True

        idx = bisect_right(self.booking, bk)
        if idx == len(self.booking) and self.booking[idx-1].end <= start:
            self.booking.append(bk)
            return True

        elif idx == 0 and end <= self.booking[0].start:
            self.booking = [bk] + self.booking
            return True
        else:
            if (self.booking[idx-1].end <= start and
                self.booking[idx].start >= end):
                self.booking.insert(idx, bk)
                return True
        return False


# My Calendar 2: https://leetcode.com/problems/my-calendar-ii/
# Insert new booking that does not cause triple booking (common to all
# 3 events)
# Solution: maintain 2 lists double bookings and single bookings
# Insert new booking, check if it overlapse with double booking
# Find overlapse in single booking, put overlapse into double booking
class MyCalendarTwo(object):
    def __init__(self):
        self.overlaps = []
        self.calendar = []

    def book(self, start, end):
        for (i, j) in self.overlaps:
            if start < j and i < end:
                return False

        for (i, j) in self.calendar:
            if start < j and i < end:
                self.overlaps.append((max(i, start), min(j, end)))
        self.calendar.append((start, end))
        return True


# HARD: My Calendar 3: https://leetcode.com/problems/my-calendar-iii/
# Insert new booking into calendar, return number of overlaped booking
# Solution: Use hashmap to store start end
# Insert new booking, if overlaps, hashmap[start] + 1, hashmap[end] + 1
# Count active intersect booking
# Optimize: use sorted array, insert keep array sorted
# Java: use treemap when insert => runtime logn
class MyCalendarThree:

    def __init__(self):
        self.calendar = defaultdict(int)
        self.times = []

    def book(self, start, end):
        self.calendar[start] += 1
        self.calendar[end] -= 1

        active = ans = 0

        for k in sorted(self.calendar):
            active += self.calendar[k]
            ans = max(ans, active)
        return ans

    def book2(self, start, end):
        insort(self.times, (start, 1))
        insort(self.times, (end, -1))
        active = ans = 0

        for (time, acc) in self.times:
            active += acc
            ans = max(ans, active)

        return ans
