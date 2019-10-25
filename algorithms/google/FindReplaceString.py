

# https://leetcode.com/problems/find-and-replace-in-string/solution/
# Find and replace string, use array to store result
def findReplaceString(S, indexes, sources, targets):
    """
    :type S: str
    :type indexes: List[int]
    :type sources: List[str]
    :type targets: List[str]
    :rtype: str
    """
    res = []
    prev = 0

    for idx, source, target in sorted(zip(indexes, sources, targets)):
        if idx > prev:
            res.append(S[prev: idx])

        if S[idx: idx + len(source)] == source:
            res.append(target)
        else:
            res.append(S[idx:idx+len(source)])
        prev = idx+len(source)
    if prev < len(S):
        res.append(S[prev:])
    return ''.join(res)
