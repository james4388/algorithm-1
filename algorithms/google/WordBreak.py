

# Word break 2
# memorize the breaking location then using dfs to return list of string
# https://leetcode.com/problems/word-break-ii/description/
def dfs_wb(s, arr, res, idx, text):
    if idx == -1:
        res.append(text[:-1])
        return

    for prev in arr[idx]:
        dfs_wb(s, arr, res, prev, s[prev+1: idx+1] + ' ' + text)


def wordBreak2(s, wordDict):
    if not s:
        return []
    n = len(s)
    arr = [[] for _ in range(n)]

    for i in xrange(n):
        for word in wordDict:
            l = len(word)
            if s[i-l+1: i+1] == word and (i - l < 0 or arr[i-l]):
                arr[i].append(i-l)
    if not arr[n-1]:
        return []
    res = []
    dfs_wb(s, arr, res, n-1, '')
    return res

print "wordBreak2....", wordBreak2('catsanddog',
                                   ['cat', 'cats', 'and', 'sand', 'dog'])
