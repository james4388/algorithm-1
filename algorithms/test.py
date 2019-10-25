def findLongestRepeatedSubstring(line):
    if not line:
        return ''

    curr = 0
    res = ''
    hashset = {}

    n = len(line)
    print("n..", n)
    for i in range(n):
        for j in range(i+1, n+1):
            sub = line[i:j]
            print("sub...", sub)
            if sub in hashset:
                if hashset[sub] <= i and j - i > curr:
                    curr = j - i
                    res = line[i:j]
            else:
                hashset[sub] = j
    return res


print("longest substring...", findLongestRepeatedSubstring("bananadnanafbanan"))
