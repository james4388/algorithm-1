

# min window string contains all string
# https://leetcode.com/problems/minimum-window-substring/description/
# Running low and high pointer, count number of character cover
# if found window, increase low pointer to shorten window size
def minWindow(s, t):
    c = Counter(t)
    need = len(t)
    n = len(s)
    lo, hi = 0, 0
    mstart, mend = -1, n
    mwidth = MAX

    while hi < n:
        char = s[hi]
        if char in c:
            if c[char] > 0:
                need -= 1
            c[char] -= 1
        # found window
        while need == 0:
            width = hi - lo + 1
            if mwidth > width:
                mwidth = width
                mstart, mend = lo, hi

            mchar = s[lo]

            if mchar in c:
                c[mchar] += 1
                if c[mchar] > 0:
                    need += 1
            lo += 1

        hi += 1

    if mstart != -1:
        return s[mstart: mend+1]
    return ""

# print "minimum window....", minWindow("ADOBECODEBANC", "ABC")
