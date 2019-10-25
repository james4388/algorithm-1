

# Expressive word
# extended group is same char with length of 3 or more, expressive words are
# word with extended group, find word in list that can be extended to be S
# https://leetcode.com/problems/expressive-words/
# Encode S and each word and compare number of char count
def expressiveWords(S, words):
    def encode(w):
        c = w[0]
        count = 1
        for i in range(1, len(w)):
            if w[i] == c:
                count += 1
            else:
                yield (c, count)
                c = w[i]
                count = 1

        yield (c, count)

    q = list(encode(S))
    ans = 0

    for word in words:
        idx = 0
        for c, num in encode(word):
            if idx >= len(q):
                break

            char, count = q[idx]
            if c != char:
                break
            if num > count or (num < count and count < 3):
                break
            idx += 1
        else:
            if idx == len(q):
                ans += 1
    return ans
