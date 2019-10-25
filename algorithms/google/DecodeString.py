

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
    stack = []
    num = ''
    char = ''
    for c in s:
        if c.isdigit():
            num += c
            if char:
                stack.append(char)
                char = ''
        elif c == '[':
            stack.append(num)
            num = ''
        elif c == ']':
            while stack:
                n = stack.pop()
                if n.isdigit():
                    char = int(n) * char
                    break
                else:
                    char = n + char
            stack.append(char)
            char = ''
        else:
            char += c
    return ''.join(stack)
