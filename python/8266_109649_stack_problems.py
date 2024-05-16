def has_valid_parens(s):
    """
    :type s: str
    :rtype: bool
    """
    left_to_right = { "{": "}","[" : "]","(" : ")" }
    stack = []
    for i in s:
        if i in left_to_right.keys():
            stack.append(i)
        elif i in left_to_right.values():
            if len(stack) == 0:
                return False
            last = stack.pop()
            if left_to_right[last] == i:
                continue
            else:
                return False
        else:
            continue

    if len(stack) == 0:
        return True
    else:
        return False


print has_valid_parens(")asdf")
print has_valid_parens("]asdf")
print has_valid_parens("}asdf")
print has_valid_parens("{a}(s)[df]")
print has_valid_parens("a[sd(fs)ad]fas{d[fsa]}")

import math
def isPowerOfThree(n):
    """
    :type n: int
    :rtype: bool
    """
    l = n
    while math.sqrt(l) % 1 != 0:
        l = math.sqrt(1)
    
    while n > 27:
        n = n / float(27)
    
    if n == 1 or n == 3 or n == 9 or n == 27:
        return True
    else:
        return False

    
print isPowerOfThree(3)
print isPowerOfThree(9)
print isPowerOfThree(27)
print isPowerOfThree(28)



