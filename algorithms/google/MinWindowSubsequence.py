from bisect import bisect_right as br
from collections import defaultdict

# Find min window W of S, so that T is subsequence of W
# Solution: - Bruteforce: 2 pointer, if S[i] == T[0], find window from i that
# has char match all T, then move pointer up until it meets another T[0]
# Runtime: O(S*T)
# Optimize: pre-process S to store index for each char in T
# Try index from first char, use binary search for next char in T
# Runtime: O(S + T + T*log(S))
def minWindowSubsequence(S, T):
    chars = set(T)
    indices = defaultdict(list)
    for idx, char in enumerate(S):
        if char in chars:
            indices[char].append(idx)
    res = ""

    for idx in indices[T[0]]:
        start = idx
        for i in range(1, len(T)):
            possibles = indices[T[i]]
            pos = br(possibles, start)
            if pos == len(possibles):
                break
            start = possibles[pos]
        else:
            if not res or start - idx + 1 < len(res):
                res = S[idx:start+1]
    return res


if __name__ == '__main__':
    print("min window subsequence...", minWindowSubsequence('acdgebbacfet', 'ace'))
