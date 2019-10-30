

# https://leetcode.com/problems/remove-comments/
# Source contains list of lines
# comment block: /*..*/ can span multiple line, and line //
# Remove comment from code
# Special case: /*/
def removeComments(source):
    res = []

    hasBlock = False
    blockStartIdx = None
    text = ''

    for num in range(len(source)):
        line = source[num]
        idx = 0
        while idx < len(line):
            chars = line[idx: idx+2]
            if hasBlock:
                if chars == '*/':
                    if (num, idx - 1) != blockStartIdx:
                        hasBlock = False
                        idx += 1
            else:
                if chars == '//':
                    break
                elif chars == '/*':
                    hasBlock = True
                    blockStartIdx = (num, idx)
                    idx += 1
                else:
                    text += line[idx]
            idx += 1

        if text and not hasBlock:
            res.append(text)
            text = ''
    if text:
        res.append(text)
    return res

source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]
print("remove comment...", removeComments(source))
