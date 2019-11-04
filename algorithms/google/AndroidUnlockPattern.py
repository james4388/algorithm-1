

# https://leetcode.com/problems/android-unlock-patterns/
# Give android pattern 3x3 find number of patterns for length in m, n
# Solution: use jump table to store require jump between non-adjacent number
# Because (1, 3, 7, 9), (2, 4, 6, 8) are symetric just need to calculate once
# and multiply to 4
class Solution:
    def numberOfPatterns(m, n):
        jumps = [[0]*10 for i in range(10)]
        # Store the number requires to make a jump e.g 1 - 3 requires 2 visited
        jumps[1][3] = jumps[3][1] = 2
        jumps[4][6] = jumps[6][4] = 5
        jumps[7][9] = jumps[9][7] = 8
        jumps[1][7] = jumps[7][1] = 4
        jumps[2][8] = jumps[8][2] = 5
        jumps[3][9] = jumps[9][3] = 6
	    jumps[1][9] = jumps[9][1] = jumps[3][7] = jumps[7][3] = 5

        visited = [False for i in range(10)]
        ans = 0

        def dfs(num, length, count):
            if length >= m:
                count += 1

            if length > n:
                return count

            visited[num] = True
            for i in range(1, 10):
                jump = jumps[num][i]
                # jump == 0 => adjacent number
                if not visited[i] and (jump == 0 or visited[jump]):
                    count = dfs(i, length + 1, count)

            visited[num] = False
            return count
        # 1, 3, 7, 9 are symetric
        ans += dfs(1, 1, 0) * 4
        # 2, 4, 6, 8 are symetric
        ans += dfs(2, 1, 0) * 4
        ans += dfs(5, 1, 0)
        return ans
