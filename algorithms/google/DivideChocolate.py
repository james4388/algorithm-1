

# https://leetcode.com/problems/divide-chocolate/
# Give chocolate bar with different sweetness, split bar into chunks, find max sweetness
# Same as split array largest sum
# tips: if left = mid, => mid = (right + left + 1)/2
class Solution:
    def maximizeSweetness(self, sweetness: List[int], K: int) -> int:
        l, r = min(sweetness), sum(sweetness) // (K + 1)

        while l < r:
            m = (l + r + 1) // 2
            count = 0
            s = 0
            for num in sweetness:
                s += num
                if s >= m:
                    count += 1
                    s = 0
            if count >= K + 1:
                l = m
            else:
                r = m - 1
        return l
