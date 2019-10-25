from collections import defaultdict


# Frog jump
# HARD: https://leetcode.com/problems/frog-jump/
# Use hash table to store next stone can be reached by set of step
def canCross(stones):
    if not stones:
        return True

    if stones[1] != 1:
        return False
    # Store next stone can jump by set of steps
    hs = defaultdict(set)
    n = len(stones)
    # Can only jump to stone 1 by 1 step
    hs[1].add(1)

    for idx in range(1, n):
        stone = stones[idx]
        print("current hs...", hs)
        if stone not in hs:
            continue

        for step in hs[stone]:
            hs[stone + step].add(step)
            hs[stone + step + 1].add(step + 1)
            if step != 1:
                hs[stone + step - 1].add(step - 1)
    return len(hs[stones[n-1]]) > 0


print("Frog jump....", canCross([0,1,3,6,10,15,16,21]))
