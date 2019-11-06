# Median of 2 sorted array
# https://leetcode.com/problems/median-of-two-sorted-arrays/
# A, B is 2 array, suppose A < B
# median A0...Ai-1 and Ai....Am
# median B0...Bj-1 and Bj....Bn
# Find i, j for A(i-1) < Bj and B(j-1) < Ai
# i + j = m + n - i - j or (m + n + 1 - i -j) => j = (m + n + 1)//2 - i
# if A(i-1) > B(j) => decrease i => increase j
# if B(j-1) > A(i) => increase i => decrease j
class MedianSolution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        A, B = nums1, nums2
        m, n = len(nums1), len(nums2)
        if m > n:
            A, B, m, n = nums2, nums1, n, m

        imin, imax = 0, m
        while imin <= imax:
            i = (imin + imax)//2
            j = (m + n + 1)//2 - i

            if i > 0 and A[i-1] > B[j]:
                imax = i - 1
            elif i < m and B[j-1] > A[i]:
                imin = i + 1
            else:
                left, right = None, None
                if i <= 0:
                    left = B[j-1]
                elif j <= 0:
                    left = A[i-1]
                else:
                    left = max(A[i-1], B[j-1])

                if (m + n) % 2 == 1:
                    return left

                if i >= m:
                    right = B[j]
                elif j >= n:
                    right = A[i]
                else:
                    right = min(A[i], B[j])

                return (left + right)/2.0
        return None
