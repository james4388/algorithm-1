

# Next closing time
# https://leetcode.com/problems/next-closest-time/description/
# e.g 11:34 ->  11:41
# Run from 1 to 24*60 to find next nearest where all number in allowed numbers
# Optimize: generate list of allowed numbers from 4 numbers
def nextClosestTime(time):
    ans = start = 60 * int(time[:2]) + int(time[3:])
    elapsed = 24 * 60
    allowed = {int(x) for x in time if x != ':'}
    for h1, h2, m1, m2 in itertools.product(allowed, repeat=4):
        hour, minute = 10*h1 + h2, 10*m1 + m2
        if hour < 24 and minute < 60:
            curr = 60*hour + minute
            curr_elapsed = (curr - start) % 24*60
            if 0 < curr_elapsed < elapsed:
                ans = curr
                elapsed = curr_elapsed
    return "{:02d}:{:02d}".format(*divmod(ans, 60))
