

# https://leetcode.com/problems/string-transforms-into-another-string/
# Give 2 string str1 and str2, return true if can transform from str1 to str2
# Solution: use hash map to store mapping char from string 1 to string 2
# if same char map to different char return false,
class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1 == str2:
            return True

        if len(str1) != len(str2):
            return False

        transforms = {}
        n = len(str1)
        for i in range(n):
            c1, c2 = str1[i], str2[i]
            if c1 not in transforms:
                transforms[c1] = c2
            else:
                prev = transforms[c1]
                if prev != c2:
                    return False
        return len(set(transforms.values())) < 26
