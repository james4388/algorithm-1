

# Phone letter combination list
# https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
def dfs_combine(m, idx, string, out):
    if idx == len(m):
        out.append(string)
        return out

    for char in m[idx]:
        dfs_combine(m, idx + 1, string + char, out)


def letterCombination(digits):
    if not digits:
        return []

    string = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
    m = [string[int(x)] for x in digits]
    out = []

    dfs_combine(m, 0, '', out)
    return out


# print letterCombination("246")
