from collections import deque


# License key formating group by k and uppercase
# https://leetcode.com/problems/license-key-formatting/description/
def licenseKeyFormatting(self, S, K):
    """
    :type S: str
    :type K: int
    :rtype: str
    """
    if not S or K <= 0:
        return ''

    buffer = deque([])
    n = len(S)
    count = 0
    for i in range(n-1, -1, -1):
        if S[i] == '-':
            continue
        buffer.appendleft(S[i])
        count += 1
        if count == K:
            buffer.appendleft('-')
            count = 0
    if buffer and buffer[0] == '-':
        buffer.popleft()

    return ''.join(buffer).upper()
