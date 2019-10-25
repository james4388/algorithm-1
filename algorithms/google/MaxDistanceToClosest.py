

# Max distance to closest person
# https://leetcode.com/problems/maximize-distance-to-closest-person/
# Use 2 pointers to find left seat and right seat occupied
# if there's no left seat, distance = right, no right seat: n - 1 - s
# else (right - left)//2
class MaxDistanceSolution:
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        if not seats:
            return 0

        m = 0
        i = 0
        n = len(seats)
        while i < n:
            while i < n and seats[i] == 1:
                i += 1
            s = i - 1
            while i < n and seats[i] == 0:
                i += 1
            e = i
            print("se...", s, e)
            if s == -1:
                m = max(m, e)
            elif e == n:
                m = max(m, n - 1 - s)
            else:
                m = max(m, (e-s)//2)
        return m
