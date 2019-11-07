

# https://leetcode.com/problems/encode-and-decode-strings/solution/
class Codec:

    def encode(self, strs):
        """Encodes a list of strings to a single string.

        :type strs: List[str]
        :rtype: str
        """
        return ''.join(len(x).to_bytes(4, byteorder='big').decode() + x for x in strs)


    def decode(self, s):
        """Decodes a single string to a list of strings.

        :type s: str
        :rtype: List[str]
        output = []
        i, n = 0, len(s)

        while i < n:
            length = int.from_bytes(bytes(s[i:i+4]), byteorder='big')
            i += 4
            output.append(s[i:i+length])
            i += length
        return output
