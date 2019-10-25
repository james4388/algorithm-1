

# Hand of straight: hand size W contains consecutive
# https://leetcode.com/problems/hand-of-straights/
# Use counter, count element, sorted item
# Loop and decrease consecutive count until has size W consecutive
def isNStraightHand(hand, W):
    """
    :type hand: List[int]
    :type W: int
    :rtype: bool
    """
    c = Counter(hand)
    for i in sorted(c):
        if c[i] > 0:
            for j in range(W, -1, -1):
                c[i + j] -= c[i]
                if c[i + j] < 0:
                    return False
    return True
