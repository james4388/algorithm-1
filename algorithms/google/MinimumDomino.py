

# https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/
# Find minimum rotation to make same upper or lower row
# Solution: because all row has same number, we can choose first domino values
# and find minimum rotation
class Solution:
    def minDominoRotations(self, A, B):
        def check(value):
            a_rotate, b_rotate = 0, 0
            for i in range(len(A)):
                if A[i] != value and B[i] != value:
                    return -1
                if A[i] != value:
                    a_rotate += 1
                if B[i] != value:
                    b_rotate += 1
            return min(a_rotate, b_rotate)

        rotations = check(A[0])
        if rotations != -1 or A[0] == B[0]:
            return rotations

        return check(B[0])
