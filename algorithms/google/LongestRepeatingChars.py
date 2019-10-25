from collections import defaultdict


# Character replacement
# https://leetcode.com/problems/longest-repeating-character-replacement/
# Find longest repeating substring created by replacing k chars
# use counter to count number of character in current window
# if number of difference exceed k, move low pointer to next character
# and substract count, re-calculate number of different chars
# Edge case: number of difference does not exceed k and pointer reach end
# => move back low pointer
# Second solution: sliding window, counter to count char, update most
# frequent char in window, diff = end - start + 1 - maxFrequent
# if diff exceeds k, move s and decrease counter,
# update length = end - start + 1
def characterReplacement(s, k):
    counter = defaultdict(int)
    count = 0
    lo, hi = 0, 1
    curr = s[0]
    ans = 1
    counter[curr] = 1

    while hi < len(s):
        if s[hi] != curr:
            count += 1

        counter[s[hi]] += 1
        if count > k:
            ans = max(ans, hi - lo)
            while lo < hi and s[lo] == curr:
                counter[s[lo]] -= 1
                lo += 1

            curr = s[lo]
            count = hi - lo - counter[curr] + 1

        hi += 1

    while count <= k and lo >= 0:
        if s[lo] != k:
            count += 1
        lo -= 1

    ans = max(ans, hi - lo - 1)

    return ans

print("character replacement...", characterReplacement('DAABBBBC', 3))
