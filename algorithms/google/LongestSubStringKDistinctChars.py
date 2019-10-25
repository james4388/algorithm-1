

# Longest substring at most k distinct chars
# https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters
# Use hashmap to store later index of characters, and count number of distinct
# characters, if count exceed k, move lower pointer, check char it passes, if
# index of character equal index in hashmap, decrease count
def longestAtMostKDistinctChars(text, k):
    if not text:
        return 0
    if k <= 1:
        return k

    low, high = 0, 0
    n = len(text)
    table = {}
    count = 0
    res = 0
    while high < n:
        char = text[high]
        if char not in table:
            count += 1
        table[char] = high
        if count > k:
            while count > k:
                lowchar = text[low]
                if low == table[lowchar]:
                    count -= 1
                    table.pop(lowchar)
                low += 1
        else:
            res = max(res, high - low + 1)
        high += 1
    return res

print "longest substring at most k distinct", longestAtMostKDistinctChars("eceded", 2)
