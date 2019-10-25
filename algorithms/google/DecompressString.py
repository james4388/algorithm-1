

# Decompress string
# similar to https://leetcode.com/problems/decode-string/description/
# a(b(c){2}){2}(b){2}d -> abccbccbbd, ((x){3}(y){2}z){2} -> xxxyyzxxxyyz
# Use stack, push string from right, if {, push current number to stack
# if } push current string to stack, if (, pop from stack, if string concat it
# if number, multiple it and push back to stack, break, if 0 concat
# all current string
# Second solution: use recursive approach, with times = 1
def decompress(pattern):
    if not pattern:
        return ''

    stack = []
    n = len(pattern)
    char_buffer = ''
    res = ''
    i = n-1
    while i >= 0:
        char = pattern[i]
        if char == '}':
            if char_buffer:
                stack.append(char_buffer)
                char_buffer = ''
            j = i
            while pattern[i] != '{':
                i -= 1
            stack.append(int(pattern[i+1:j]))
        elif char == ')':
            char_buffer = ''
        elif char == '(':
            while stack:
                val = stack.pop()
                if isinstance(val, int):
                    char_buffer = char_buffer * val
                    break
                else:
                    char_buffer = char_buffer + val
            stack.append(char_buffer)
            char_buffer = ''
        else:
            char_buffer = char + char_buffer
        i -= 1
    stack.append(char_buffer)
    while stack:
        res += stack.pop()
    return res
