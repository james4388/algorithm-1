


# Find next closest time of current given time
# Solution: calculate in minute: m = hour * 60 + min, increase value and check
# if all number in allowed number, runtime: O(24*60)
class Solution(object):
    def nextClosestTime(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            if all(digit in allowed
                    for block in divmod(cur, 60)
                    for digit in divmod(block, 10)):
                return "{:02d}:{:02d}".format(*divmod(cur, 60))


    def nextClosestTime(self, time: str) -> str:
        if time == '00:00':
            return time

        hour, minute = time.split(':')
        arr = list(itertools.product(hour + minute, repeat=2))
        ans = '24:60'
        _min = '24:60'
        for h1, h2 in arr:
            for m1, m2 in arr:
                if h1 > '2' or (h1 == '2' and h2 > '3') or m1 > '5':
                    continue
                ts = '{}{}:{}{}'.format(h1, h2, m1, m2)
                if time < ts < ans:
                    ans = ts
                _min = min(_min, ts)
        return ans if ans != '24:60' else _min
