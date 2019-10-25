

# https://leetcode.com/problems/isomorphic-strings/
# Isomorphic string, use hash map to map from one char to other
def isIsomorphic(s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    if len(s) != len(t):
        return False
    m = {}
    for i in range(len(s)):
        if s[i] in m:
            if t[i] != m[s[i]]:
                return False
        else:
            m[s[i]] = t[i]
    # Check if 2 different chars map to same char
    vals = m.values()
    if len(vals) != len(set(vals)):
        return False
    return True
