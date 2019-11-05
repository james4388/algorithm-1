from collections import defaultdict
from bisect import bisect_right


# https://leetcode.com/problems/shortest-way-to-form-string/
# Give source and target string, return how many source subsequence can add
# to form target
# Solution: compare char in target to source, if match increase index, if over
# source increase count, if not in set return -1
# Optimize:
# - Space: instead of use set, we can keep track of prev index, if after compare
# index not increase, => cannot find char
# - Runtime: O(M*N) if pre-process source to have char index: e.g a -> [1, 4, 7]
# we can use binary search for target char, => O(N*logM)
# - If small char set, we could pre-compute next position for every index
# e.g apple for p -> [1, 2, -1, -1, -1] => O(M + N)
def shortestWay(source: str, target: str) -> int:
    if source == target:
        return 1

    chars = set(source)
    count = 0
    idx = 0
    while idx < len(target):
        if target[idx] not in chars:
            return -1

        count += 1
        for char in source:
            if char == target[idx]:
                idx += 1
            if idx >= len(target):
                break
    return count


def shortestWayOpt1(source, target):
    if source == target:
        return 1

    count = 0
    idx = 0
    while idx < len(target):
        count += 1
        org = idx
        for char in source:
            if char == target[idx]:
                idx += 1
            if idx >= len(target):
                break

        if idx == org:
            return -1
    return count


def shortestWayOpt2(source, target):
    indices = defaultdict(list)

    for idx, char in enumerate(source):
        indices[char].append(idx)

    count = 0
    pos = -1
    for char in target:
        if char not in indices:
            return -1
        pos = bisect_right(indices[char], pos)
        if pos == len(indices[char]):
            pos = -1
            count += 1
    return count


def shortestWayOpt3(source, target):
    count = 0
    indices = {}
    n = len(source)

    for idx, char in enumerate(source):
        if char not in indices:
            indices[char] = [0 for i in range(n)]
        indices[char][idx] = idx + 1

    for char in indices:
        prev = 0
        pos = indices[char]
        for i in range(n - 1, -1, -1):
            if pos[i] != 0:
                prev = pos[i]
            else:
                pos[i] = prev

    pos = 0
    for char in target:
        if not indices.get(char):
            return -1
        arr = indices[char]
        pos = arr[pos]
        if pos == 0 or pos == n:
            count += 1
            pos = 0
    return count
