

# Return number of delete to ensure count of each character are unique
def uniqueAppreance(S):
    counter = [0 for i in range(26)]

    for char in S:
        counter[ord(char) - ord('a')] += 1

    _max = max(counter)
    allocation = [[] for i in range(_max+1)]


# Change each ? to character so that there's no same char continuously
# e.g: a?bv?c
def replaceString(S):
    pass
