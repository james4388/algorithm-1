'''
68) https://leetcode.com/problems/text-justification
- Text justification: calculate len of current word  +
number of word > maxWidth then stop, padding more space and add to result
'''

# https://leetcode.com/problems/text-justification/
def fullJustify(words, maxWidth):
    def buildStr(w, numspace):
        if len(w) == 1:
            return w[0] + " " * numspace
        # Space per word
        space = numspace // (len(w) - 1)
        # Extra space for left
        extra = numspace % (len(w) - 1)
        ans = ""
        for i in range(len(w) - 1):
            ans += w[i] + " " * space
            if extra > 0:
                ans += " "
                extra -= 1

        ans += w[-1]
        return ans

    last = 0
    n = len(words)
    size = 0
    ans = []

    for idx in range(n):
        if size + len(words[idx]) + idx - last > maxWidth:
            ans.append(buildStr(words[last: idx], maxWidth - size))
            size = 0
            last = idx
        size += len(words[idx])
        if idx == n - 1:
            ans.append(" ".join(words[last: n]) + " " * (maxWidth - size - idx + last))
    return ans


if __name__ == '__main__':
    words = ["What","must","be","acknowledgment","shall","be"]
    # words = ["hello", "my", "name", "is", "coding", "!"]
    print(fullJustify(words, 16))
