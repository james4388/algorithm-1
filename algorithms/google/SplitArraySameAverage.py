

# HARD: https://leetcode.com/problems/split-array-with-same-average/
# Brute force: generate all subset sum and check if 2 average are equal
# Math: call s, length is total sum and length of array
# 2 sum: s1, s2 with length n1, n2 => s1/n1 == s2/n2 or s1*n2 == s2*n1
# s1 + s2 = s, n1 + n2 = n => s1 * (len - n1) == (s - s1) * n1
# Optimize: observe that with if 2 pair of same sum and number of elements
# we dont need to re-calculate again, => use hashset to store
# (current_sum, num)
class SplitArraySameAverage(object):
    def split(self, visited, arr, totalSum, length, currSum, num, idx):
        if (num != 0 and num != length and
            currSum * (length - num) == (totalSum - currSum) * num):
            return True

        if (currSum, num) in visited:
            return False

        for i in range(idx, length):
            if self.split(arr, totalSum, length,
                          currSum + arr[i], num + 1, i + 1):
                return True
            else:
                visited.add((currSum + arr[i], num + 1))
        return False

    def splitArraySameAverage(self, A):
        if not A:
            return True
        visited = set()
        s, length = sum(A), len(A)
        return self.split(visited, A, s, length, 0, 0, 0)

    # Assume split into 2 array B, C and B is smaller => B <= len(A) / 2
    # sumB / lenB = (sumA - sumB) / (lenA - lenB) => sumA / lenA = sumB / lenB
    # => sumB = sumA * lenB / lenA
    # => sumA * lenB % lenA == 0 (natural number)
    # Knapsack problem, for each number we could choose to add or not into set
    # Run time: sum = M * n, 2 loops = n ^ 2 => O(n^3 * M)
    # M is max value of each element
    def splitArraySameAverage2(self, A):
        m = len(A) // 2
        s = sum(A)

        dp = [[0 for _ in range(m + 1)] for _ in range(s + 1)]
        dp[0][0] = 1

        for num in A:
            for i in range(s, num - 1, -1):
                for j in range(1, m + 1):
                    dp[i][j] = dp[i][j] or dp[i - num][j - 1]

        for i in range(1, m + 1):
            if s * i % len(A) == 0 and dp[s * i % len(A)][i]:
                return True
        return False


sol = SplitArraySameAverage()
arr = [3863,703,1799,327,3682,4330,3388,6187,5330,6572,938,6842,678,9837,8256,6886,2204,5262,6643,829,745,8755,3549,6627,1633,4290,7]
sol.splitArraySameAverage(arr)
