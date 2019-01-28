'''
499)
- The maze 3
- HARD: https://leetcode.com/problems/the-maze-iii/
- Ball can only change direction if it hits the wall

- Use BFS: check for possible directions of balls, add into queue
 (x, y, direction, step), pop an item from queue, then move it in direction
until next wall and increase step, select next direction so that ball not going
back, while moving ball if it drop into hole remove it and compare number of
step, if less than current or equal and smaller lexicographically

'''


'''
732) https://leetcode.com/problems/my-calendar-iii/
- My calendar 3: return K largest intersect bookings in booking list

- when insert new booking, calculate the maximum intersection of each
interval, need data structure to easy calculate number of intersections
- Use 2 sorted array to store start and end time, count start time +1 and end
time -1 => n^2 * logn time to sort and n to find max intersection, use
bisect.insort => n^2
- Use hashmap table, table[start] += 1, table[end] -= 1, count sorted array
'''

'''
489) https://leetcode.com/problems/robot-room-cleaner/
- Robot cleaner room: design algorithm for robot to clean the room with
obstacle, robot can move, turnLeft, turnRight and clean

- Use DFS move until it cannot move anymore, then backtrack to
original location using (turnLeft, turnLeft, move, turnLeft, turnLeft);
then turn into other direction using turnRight (4 total directions),
use set to keep track of cleaned cell.
'''

'''
493) https://leetcode.com/problems/reverse-pairs/
=> similar to: 315) https://leetcode.com/problems/count-of-smaller-numbers-after-self/
- Reversed pairs: i < j and nums[i] > 2*nums[j]

- Bruteforce: for each position i, run j = i + 1 and count pair for nums[i] >
2*nums[j] => runtime O(n^2)
- Need data structure to store 2*nums[j] for faster count number of pair less
than nums[i] => self balance tree for insert => nlogn
- Use segment tree or binary indexed tree: create tree with empty value,
sort array with its rank, run from right to left, each time update at its
index value = 1 into tree, count = sum(0, index - 1). Runtime:
create tree O(n), update and sum take log(n) for each number => O(nlogn)
- Use BST: node has count_greater, insert a node into BST, update count along
insert path if root less than insert value, then search for values 2*nums[j]
and add count_greater along the path
'''

'''
803) https://leetcode.com/problems/bricks-falling-when-hit/
- Brick falling when hits

- Set hit in grids, if it's 1 => 0, 0 => -1, run dfs from top and mark no
dropping bricks (value=2), get grid after all hits, reversely add hit bricks
back if it's brick 1 and connected (one of surround cell = 2),
then run dfs to mark new no-drop brick and return the amount
'''

'''
135) https://leetcode.com/problems/candy/
- Candy problem: each children has rating, higher rating children receive more
candy than its neighboors (left and right)
- Run from left to right, compare rating to left side, if higher then assign
candy = left_candy + 1, do same thing from right to left, result is total candy

'''


'''
248) https://leetcode.com/problems/strobogrammatic-number-iii/
- Medium: n = 1 -> 0, 1, 8 look the same after rotate 180, 6 -> 9 and 9 -> 6
- recursive call for n - 2, for possible 00, 11, 69, 88, 96
- Hard: Add all pairs with length from low to high and filter them
- Optimize: DFS for smaller numbers
'''

'''
753) https://leetcode.com/problems/cracking-the-safe/
- Cracking the safe, n = 2, k = 2, => 01100
- Solution: k^n-1 nodes at each node has k values, try to combine all possible
values, and find shortest string, next values should overlap with n - 1 char
e.g n = 3, k=3,  011 -> next 112, use set to store visited values, if number of
visited values = k ^ n, return true and stop processed.
'''

'''
149) https://leetcode.com/problems/max-points-on-a-line/
- Max point on a line: calculate slope: (y1-y2)/(x1-x2), and store in hashmap
for pair with same slope
'''

'''
188) https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
- Best time to buy and sell stock at most k transactions
- dp[k][n] is maximum profit k transaction and up to day n

- If we made a sell at day n, then we can make 1 transaction with maximum
profit = max(prices[n] - prices[x]) for x in 0 to n-1,
=> total profit = max(dp[k-1][x] + prices[n] - prices[x])

- If we don't make a sell at day n, then dp[k][n] = dp[k][n-1]

- Combine both conditions:
- dp[k][n] = max(dp[k][n-1], dp[k-1][x] + prices[n] - prices[x]) (x in 0...n-1)
= max dp[k][n-1], prices[n] + max( dp[k-1][x] - prices[x])
=> dimension: dp[k+1][n+1], reduce to dp[k+1][n] because prices[n-1]
- Normal run: 3 loops i = 1 to k; j = 1 to n; x = 0 to j
- Optimize: we need to keep track of maximum dp[k-1][x] - prices[0], dont need
to run for every j again because x in 0...j, store localMax dp[i-1][j] - prices[j]
when j is running and update this value.
- Base case: dp[0][n] = 0; dp[k][0] = 0
'''

'''
Regex pattern matching:
10) https://leetcode.com/problems/regular-expression-matching/description/
- DP solutions:
- s[i] = p[j] or p[j] = '.': dp[i][j] = dp[i-1][j-1]
- p[j] = '*': 2 cases
- p[j-1] != s[i]: dp[i][j] = dp[i][j-2]
- p[j-1] == s[i] or p[j-1] = '.':
    dp[i][j] = dp[i-1][j] (match multiple)
    or dp[i][j-1] (single match)
    or dp[i][j-2] (empty)
- Base case: dp[0][j] = dp[0][j-2] if p[j-1] == '*'
'''

'''
Prefix and suffix search
745) https://leetcode.com/problems/prefix-and-suffix-search/
- Modify trie node to indicate it's prefix or suffix, search both prefix and
suffix chars at the same time
- if not prefix or suffix is empty, we only need to search for one
- store weight at each node
'''

'''
97) https://leetcode.com/problems/interleaving-string/
- Interleave string: s1 combine s2 interleave to become s3
- Compare length s1 + length s2 = s3
- Dynamic programing: x, y, z are length of s1, s2 and s3 z = x + y
- dp(z, x, y) = dp(z - 1, x - 1, y) or dp(z-1, x, y - 1)
- for 2 cases: s3[z] = s1[x] or s3[z] = s2[y]
- dp[i][j] = (dp[i][j-1] and s2[j-1] == s3[i+j-1]) or (dp[i-1][j] and s1[i-1] == s3[i+j-1])
'''


'''
316) https://leetcode.com/problems/remove-duplicate-letters/
- Remove duplicate in string for to have smallest lexicographical
- Use array 26, add index of each characters
- Find last appear of each character, loop from begining find the smallest index
- From begining to smallest index, find the smallest character and put into result
- Update the begin to smallest character
'''

'''
428) https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/
- Serialize n-ary tree: Use a queue and recursively serialize child into queue,
padding special char e.g '#' to indicate end of children list
'''

'''
715) https://leetcode.com/problems/range-module/
- Range module: remove, add and query range
- insert range left, right into sorted structure [-infinite, +infinite]
- query: find index of left and right using bisect and bisect left, index
always even
'''

'''
727) https://leetcode.com/problems/minimum-window-subsequence/discuss/188265/Python-O(T*log(S))-Solution
- Minimum window substring: store indices of each character
'''

'''
115) https://leetcode.com/problems/distinct-subsequences/
- String s and t: x1...xi and y1...yj
- if xi == yj: dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
- xi != yj: dp[i][j] = dp[i-1][j]
- base case: dp[i][0] = 1, dp[0][j] = 0, dp[0][0] = 1
'''

'''
815) https://leetcode.com/problems/bus-routes/
- Find minimum bus to reach from start to end
- Make edge from bus i to bus j if there's connected station
- Find start station and end station in which bus routes
- Do BFS and count how many different bus route from start station bus
'''

'''
679) https://leetcode.com/problems/24-game/
- DFS for 3 operations, choose 2 of them for +, -, *, / and get result append
'''

'''
354) https://leetcode.com/problems/russian-doll-envelopes
- RussianDoll envelop: find maximum envelop can fit into size
- sort envelopes, loop through sorted list, use binary search to find insert
point
'''

'''
403) https://leetcode.com/problems/frog-jump/
- Frog jump: k - 1, k, k + 1 step from previous jump
- Use hash table to store previous jump to reach current stone
- Find out which stone it can reach next for 3 ways jump
'''

'''
68) https://leetcode.com/problems/text-justification
- Text justification: calculate len of current word + previous length +
number of word > maxWidth then stop, padding more space and add to result
'''
