

# https://leetcode.com/problems/new-21-game/
# Alice has 0  point, can draw from 1...W, until has K or more points
# return probability to has N or less points
# Solution: W = 3
# prob(5) = prob(4) * 1/3 + prob(3) * 1/3 + prob(2) * 1/3
# p(x) = (p(x-1) + p(x-2) + ... + p(x-w)) / w
# Sliding window: maintain sum = p(x) if x > w => sum -= p(x-w-1)
# p(0) = 1.0

class Solution:
    def new21Game(self, N: int, K: int, W: int) -> float:
        if N >= K + W:
            return 1.0

        probs = [0 for i in range(N+1)]
        probs[0] = 1.0
        wsum = 1.0
        for x in range(1, N+1):
            probs[x] = wsum/W
            # Because game stop at K so W[k+1] cannot = W[k] + prob(1)
            if x < K:
                wsum += probs[x]
            if x - W >= 0:
                wsum -= probs[x - W]
        return sum(probs[K:])



if __name__ == '__main__':
    s = Solution()
    print(s.new21Game(21, 17, 10))
