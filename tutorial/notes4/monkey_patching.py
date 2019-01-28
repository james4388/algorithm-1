''' Monkey patching is making variable assignments into someone else's namespace.

Legitimate use cases for monkey patching:

    * Fixing erronenous constants
    * Improving error messages
    * Adding debugging capability (add in input/output type checking, range checks etc)
    * Making functions more robust (handle a broader range inputs)

IIlegitimate cases:

    * If you ever monkey patch your own code, you're living in a state of sin!
    * You should fix your code directly

'''

import algebra
import math

algebra.pi = math.pi                              # Monkey patch

orig_area_triangle = algebra.area_triangle        # Step 1: Save original

def better_area_triangle(base, height):           # Step 2: Write a wrapper
    'Wrap the algebra package to supply useful and correct error messages'
    try:
        orig_area_triangle(base, height)
    except RuntimeError:
        raise ValueError('Negative base and height not supported, use positive inputs instead')

algebra.area_triangle = better_area_triangle      # Step 3: Monkey patch

orig_sqrt = math.sqrt                             # Step 1: Save original

def better_sqrt(x):                               # Step 2: Write a wrapper
    'Wraps the math.sqrt function to add support for negative inputs'
    if x >= 0.0:
        return orig_sqrt(x)
    return orig_sqrt(-x) * 1j

math.sqrt = better_sqrt                           # Step 3: Monkey patch

if __name__ == '__main__':
    print u'My sources tell me that \N{greek small letter pi} =', algebra.pi
    print 'And the area of circle of radius ten is:', algebra.area(10)

    try:
        print 'The area of the 1st triangle is', algebra.area_triangle(10, 20)
        print 'The area of the 2nd triangle is', algebra.area_triangle(-10, 20)
    except ValueError:
        print 'Oops, sorry about the negative inputs'

    print 'Solutions to 12x^2 + 23x + 10 = 0 are:'
    print algebra.quadratic(a=12, b=23, c=10)
    print 'Solutions to 12x^2 + 5x + 10 = 0 are:'
    print algebra.quadratic(a=12, b=5, c=10)
