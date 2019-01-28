__version__ = '1.0'

import os

# String algorithm
"""
Reformating license number problem:
Given license string with invalid format and given length k
of each string group separated by dash '-'
"""
def license(str, k):
    count = 0
    output = ''
    for i in xrange(len(str)-1, -1, -1):
        if str[i] != '-':
            count += 1
            output = str[i].upper() + output
        if count == k and i != 0:
            count = 0
            output = '-' + output
    return output


"""
Find indices split array into 2 equal sum
"""
def equal_sum(l):
    s = sum(l)
    q = 0
    for i in range(len(l)):
        p = s - l[i] - q
        if p == q:
            return i
        q += l[i]
    return -1

from collections import deque
def solution(A, B, K):
    g = {}
    l = len(A)
    stack = deque([])
    for i in range(l):
        g[A[i]] = []
        g[A[i]].append(B[i])
    for j in range(l):
        if g.get(B[j], None) is not None:
            g[B[j]].append(A[j])
        else:
            g[B[j]] = [A[j]]
    items = g.items()
    while items:
        e = items.pop()
        stack.append(e[0])
        stack.extend(e[1])
        
"""
    Given a set of ranges e.g S = {(1,4), (30,40), (20, 91), (8, 10), (6,7), 
    (3,9), (9, 12), (11, 14)} and a given range (3, 13). Write algorithm 
    to find smallest set of range that covers the target range. All of the range
    must be overlap in order to be considered as spanning the entire 
    target range. e.g {(3,9), (9,12), (11,14)}
"""        
        
def range_cover(rset, r):
    pass
    
    
"""
    Given a list of integers, find the highest obtainable value by 
    concatinating these together
    e.g: 9, 918, 917 would give 9918917
    but 1, 112, 113 give 1131121
"""
def highest_obtain(vars):
    def compare(x, y):
        x = str(x)
        y = str(y)
        i = len(x)
        j = len(y)
        maxl = i if i > j else j
        for k in range(0, maxl):
            l = x[k%i]
            m = y[k%j]
            if l > m:
                return 1
            if l < m:
                return -1
        return 0
        
    vars.sort(compare, reverse=True)
    l = [str(x) for x in vars]
    return "".join(l)


def isSubstring(s1, s2):
    """docstring for isSubstring
        Check if one string is substring of other
    """
    return s2 in s1

def isRotation(s1, s2):
    """docstring for isRotation
        Check if one string is rotation of other
    """
    l = len(s1)
    if len(s2) == len(s1) and l > 0:
        strs = s1 + s1
        return isSubstring(strs, s2)
    return False

# Tree algorithm
class TreeNode:
    left = None
    right = None
    parent = None
    value = 0
    
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent

def checkHeight(n):
    if not n:
        return 0
    
    left = checkHeight(n.left)
    if left == -1:
        return -1
    
    right = checkHeight(n.right)
    if right == -1:
        return -1
    
    if abs(left - right) > 1:
        return -1
    
    return max(left, right) + 1
            
def isBalanced(n):
    """docstring for isBalanced"""
    if checkHeight(n) == -1:
        return False
    return True

# Create minimal BST from sorted array
def buildBST(arr, left, right):
    if right < left:
        return None
    
    if right == left:
        return TreeNode(arr[left])
         
    mid = (left + right)/2
    n = TreeNode(arr[mid])
    n.left = buildBST(arr, left, mid - 1)
    n.right = buildBST(arr, mid + 1, right)
    return n

def createMinimalBST(arr):
    return buildBST(arr, 0, len(arr) - 1)


# Beautiful arrangement
# Given N find number of arrangement
# satify ith number divisable to i
# or i divisable to ith number
def countDivisibleNum(n):
    count = 0
    for i in range(n-1, 0, -1):
        if n%i == 0:
            count += 1
    return count
    
def beautifulArrange(n):
    l = [countDivisibleNum(x) for x in range(n, 1, -1)]
    return sum(l) + 1

# Find highest difference
def highestDifference(arr):
    val = 0
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if arr[i] - arr[j] > val:
                val = arr[i] - arr[j]
    return val

def highestDifference2(arr):
    m = arr
    idx = 0
    val = 0
    while len(m) > 0:
        n = max(m)
        i = arr.index(n, idx)
        for j in range(i, -1, -1):
            if n - arr[j] > val:
                val = n - arr[j]
        m = arr[i+1:]
        idx = i + 1
    return val
# Compress a string
def compress(s):
    if not s:
        return ''
        
    current = s[0]
    count = 1
    result = ''
    i = 1
    while i < len(s):
        if s[i] == current:
            count += 1
        else:
            result += str(count) + current
            count = 1
            current = s[i]
        i += 1
    result += str(count) + current
    return result

# Find sub 2 dimensions array
def isSubArray(s, p, roff, coff, rp, cp):
    for i in xrange(rp):
        for j in xrange(cp):
            if p[i][j] != s[roff+i][coff+j]:
                return False
    return True
    
def findSubArray(s, rs, cs, p, rp, cp):
    if rs < rp or cs < cp:
        return False
    
    for i in xrange(rs - rp + 1):
        for j in xrange(cs - cp + 1):
            if s[i][j] == p[0][0]:
                if isSubArray(s, p, i, j, rp, cp):
                    return True
    return False

# Check funny string
def isFunnyStr(s):
    n = len(s)
    for i in range(1, n/2 + 1):
        d1 = ord(s[i]) - ord(s[i-1])
        d2 = ord(s[n-1-i]) - ord(s[n-i])
        if abs(d1) != abs(d2):
            return False
    return True

# Check pangram string
def isPangram(s):
    d = s.lower()
    en = 'abcdefghijkmnopqrstxyz'
    for i in en:
        if not i in d:
            return False
    return True

# Pascal triangle at line i
# optimize to use O(k) space
def getPascalRow(rowIndex):
    m = []
    for i in xrange(rowIndex + 1):
        m.append(1)
        for j in xrange(i - 1, 0, -1):
            m[j] += m[j-1]
    return m

# Find sublist of list non-duplicate
def subList(nums):
    if not nums:
        return []
        
    result = [[]]
    for i in nums:
        m = []
        for j in result:
            l = j + [i]
            if l not in result:
                m.append(l)
        result.extend(m)
    return result

# Find single number in duplicate array
def findSingle(nums):
    if not nums:
        return None
    r = 0
    for i in nums:
        r ^= i
    return r

# Count sum range
def countRangeSum(self, nums, lower, upper):
    first = [0]
    for num in nums:
        first.append(first[-1] + num)
    def sort(lo, hi):
        mid = (lo + hi) / 2
        if mid == lo:
            return 0
        count = sort(lo, mid) + sort(mid, hi)
        i = j = mid
        for left in first[lo:mid]:
            while i < hi and first[i] - left <  lower: i += 1
            while j < hi and first[j] - left <= upper: j += 1
            count += j - i
        first[lo:hi] = sorted(first[lo:hi])
        return count
    return sort(0, len(first))
    
# Power of 2, 3
def pofTwo(n):
    return n & (n-1) == 0

def pofThree(n):
    if n <= 0:
        return False
    
    while n > 1:
        if n%3 != 0:
            return False
        n /= 3
    return True

# Check if string is palindrome
# ask: if empty string, string contains only non-alpha chars
def isPalidrome(s):
    if not s or not s.strip():
        return True
    
    s1 = s.lower()
    chars = [x for x in s1 if x.isalnum()]
    return chars == chars[::-1]
    

# Addictive number
from collections import deque
def checkValidNum(first, second, remain):
    if len(remain) == 0:
        return True
    l = remain    
    t = len(remain) - 1
    while t >= 0:
        u, v = len(first) - 1, len(second) - 1
        out = deque([])
        comp = 0
        #print "first={}, second={}".format(first, second)
        while u >= 0 or v >= 0:
            n = ord(first[u]) - ord('0') if u >=0 else 0
            m = ord(second[v]) - ord('0') if v >= 0 else 0
            k = (m + n + comp)%10
            comp = (m + n + comp)/10
            out.appendleft(str(k))
            u -= 1
            v -= 1
            
        if comp > 0:
            out.appendleft(str(comp))
            
        o = "".join(out)
        #print "o={}, l={}".format(o, l)
        if not l.startswith(o):
            return False
        first = second
        second = o
        l = l[len(o):]
        t -= len(o)
    return True
        
    
def isAddictive(s):
    l = len(s)/3
    for i in xrange(1, l+1):
        first = s[:i]
        for j in xrange(i+1, l*2 + 1):
            second = s[i:j]
            if checkValidNum(first, second, s[j:]):
                return True
    return False

#print isAddictive("112358")
#print isAddictive("199100199")
#print isAddictive("1203")

# Maximum product sub array
# Keep 2 local maximum and minimum
# multiply number k, if k*lmax or k*lmin > lmax -> update lmax else lmax->k
# k*lmax or k*lmin < lmin -> update lmin else lmin->k
# update global max
def maxSubArray(nums):
    if not nums:
        return 0
        
    if len(nums) == 1:
        return nums[0]
        
    lmax, lmin = 0, 0
    res = 0
    for num in nums:
        p, q = num*lmax, num*lmin
        lmax = max(num, p, q)
        lmin = min(num, p, q)
        res = max(res, lmax)
    return res

#print maxSubArray([2, 3, 4, -1, 3, 5, 2, -2])

def findSubsetSum(arr, k):
    """
        Backtracking technique
        arr: list[int]
        k: int
        return list[list]
    """
    if not arr:
        return []
    
    result = []
    arr.sort()
    for i in range(len(arr)):
        if arr[i] == k:
            result.append([arr[i]])
        elif arr[i] < k:
            re = findSubsetSum(arr[i+1:], k - arr[i])
            result.extend([[arr[i]] + x for x in re if x])
        else:
            break
    return result

#print findSubsetSum([3, 1, 4, 7, 5, 10], 11)

# Find missing number in two list
def findMissing(p, q):
    tmp = [0 for x in xrange(101)]
    m = min(q)
    for i in q:
        tmp[i - m] += 1
    for j in p:
        tmp[j - m] -= 1
    result = [str(k+m) for k, v in enumerate(tmp) if v > 0]
    print " ".join(result)

# Maximum stock profit 
def maxProfit(stocks):
    if len(stocks) <= 1:
        return 0
    profit = 0
    m = stocks[-1]
    for i in xrange(len(stocks) - 2, -1, -1):
        if m > stocks[i]:
            profit += m - stocks[i]
        m = max(m, stocks[i])
    return profit
# Fizzbuzz fibonacci
import math
def isFibonacci(n):
    if n == 0:
        return True
    p = math.sqrt(5*n*n - 4)
    q = math.sqrt(5*n*n + 4)
    return p%1 == 0 or q%1 == 0

# Find longest palindrome
def longestPalindrome(self, s):
    lenS = len(s)
    if lenS <= 1: return s
    minStart, maxLen, i = 0, 1, 0
    while i < lenS:
        if lenS - i <= maxLen / 2: 
            break
            
        j, k = i, i
        while k < lenS - 1 and s[k] == s[k + 1]: 
            k += 1
            
        i = k + 1
        while k < lenS - 1 and j and s[k + 1] == s[j - 1]:  
            k, j = k + 1, j - 1
            
        if k - j + 1 > maxLen: 
            minStart, maxLen = j, k - j + 1
            
    return s[minStart: minStart + maxLen]

# sort by frequency
from operator import itemgetter, attrgetter
def sortFreq(arr):
    hm = {}
    for num in arr:
        if num in hm:
            hm[num] += 1
        else:
            hm[num] = 1
    #print hm
    l = hm.items()
    l.sort(key=itemgetter(1, 0))
    res = []
    for (k, v) in l:
        res.extend([k]*v)
    return res

#print sortFreq([1, 3, 2, 5, 4, 2])

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]

# l = all_perms([2,3,4,5])
# for i in l:
#     print i

# Buy and sell stock
def sellStock(arr):
    if not arr or len(arr) <= 1:
        return [-1, -1]
    sell_idx = -1
    sell_max = -1
    val = 0
    buy_idx = -1
    
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] > sell_max:
            sell_max = arr[i]
            sell_idx = i
        else:
            if sell_max - arr[i] > val:
                val = sell_max - arr[i]
                buy_idx = i
    return [buy_idx, sell_idx]

#print sellStock([5, 0, 4, 2, 3, 8])
            
    
    


    
    