

# https://leetcode.com/problems/decode-string/description/
# example 3[a2[c]e]2[df] -> acceacceaccedfdf
# use stack, push char into stack, if [ append current num, if ] pop stack
# if pop item is number, then multiple char, if not, concatinate item to
# current char, append char to stack
def decodeString(s):
    """
    :type s: str
    :rtype: str
    """

    currStr = ""
    currNum = 0
    arr = []
    
    for char in s:
        if char == '[':
            arr.append(currStr)
            arr.append(currNum)
            currStr = ""
            currNum = 0
        elif char == ']':
            num = arr.pop()
            prevStr = arr.pop()
            currStr = prevStr + num * currStr
        elif char.isdigit():
            currNum = currNum * 10 + int(char)
        else:
            currStr += char
    return currStr
