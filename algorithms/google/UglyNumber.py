

# Find Nth ugly number
# https://leetcode.com/problems/ugly-number-ii
# First number 1, next number: min(1*2, 1*3 or 1*5) = 2
# next number = 2*2, 1*3, 1*5 = 3; next = 2*2, 2*3, 1*5 = 4
# Use index2, index3, index5 to store next index for number need to multiple
# to 2, 3, 5
def findNthUglyNumber(n):
    ugly = [1]
    index2 = index3 = index5 = 0
    for k in range(1, n):
        val = min(ugly[index2] * 2, ugly[index3] * 3, ugly[index5] * 5)
        if val == ugly[index2] * 2:
            index2 += 1
        if val == ugly[index3] * 3:
            index3 += 1
        if val == ugly[index5] * 5:
            index5 += 1
        ugly.append(val)
    return ugly[-1]


# Super ugly number
# https://leetcode.com/problems/super-ugly-number/description/
# Solve as ugly number, handle duplicate by increasing index of prime, if it's
# smaller than current ugly number
def nthSuperUglyNumber(self, n, primes):
    """
    :type n: int
    :type primes: List[int]
    :rtype: int
    """
    if n <= 0:
        return -1

    if n == 1:
        return 1

    m = len(primes)
    ugly = [1]
    indices = [0 for x in range(m)]
    _max = float('inf')
    for i in range(1, n):
        idx = 0
        curr = _max
        for j in range(m):
            val = ugly[indices[j]] * primes[j]
            # Avoid duplicate by increasing index of prime j
            if val <= ugly[-1]:
                indices[j] += 1
                val = ugly[indices[j]] * primes[j]
            if val < curr:
                curr = val
                idx = j
        ugly.append(curr)
        indices[idx] += 1

    return ugly[-1]
