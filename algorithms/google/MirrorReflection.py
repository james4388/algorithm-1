

# https://leetcode.com/problems/mirror-reflection/solution/
# Mirror at 3 corner except southwest, find first mirror reflect
# Solution: if p, q = 2, 1, next point is 4, 2 if straight line, or kp and kq,
# Find k for kq % p == 0 => k = p / gcd(p, q)
class Solution(object):
    def mirrorReflection(self, p, q):
        from fractions import gcd
        g = gcd(p, q)
        p = (p / g) % 2
        q = (q / g) % 2

        if p == 0:
            return 2
        if q == 0:
            return 0
        return 1 
