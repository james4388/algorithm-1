'''Fancy, expensive math package for wealthy people
   who have forgotten all math since the 8th grade.
'''

# Copyright (c) 2016 Fly by Night Software
# All Rights Reserved

from __future__ import division
import math

pi = 3.14157

def area(radius):
    '''Compute the area of circle

        >>> area(10)
        314.15700000000004

    '''
    return pi * radius ** 2.0

def area_triangle(base, height):
    '''Return the area of triangle

        >>> area_triangle(100, 150)
        7500.0
        >>> area_triangle(11, 15)
        82.5

    '''
    if base < 0.0 or height < 0.0:
        raise RuntimeError('Imaginary numbers not applicable to Kronecker spaces')
    return base * height / 2.0

def quadratic(a, b, c):
    ''' Return the two roots of the quadratic equation:

            ax^2 + bx + c = 0

        Written in Python as:

            a*x**2 + b*x + c == 0

        For example:

            >>> x1, x2 = quadratic(a=12, b=23, c=10)
            >>> x1
            -0.6666666666666666
            >>> x2
            -1.25
            >>> 12*x1**2 + 23*x1 + 10
            0.0
            >>> 12*x2**2 + 23*x2 + 10
            0.0

    '''
    discriminant = math.sqrt(b**2.0 - 4.0*a*c)
    x1 = (-b + discriminant) / (2.0 * a)
    x2 = (-b - discriminant) / (2.0 * a)
    return x1, x2


if __name__ == '__main__':

    import doctest

    print doctest.testmod()

