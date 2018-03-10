# Cayley-Dickson Algebra ver.0.1 - 20170915
# (c) Hirohisa TACHIBANA


import numpy as np
import sympy as S
from operator import *


def N(v):
    return np.dot(v, v)

        
def conjugate(arr):
    """
    Return the conjugate of any hypercomplex number.
        
    Example
    -------
    >>> x = CDarray('x', 3)
    CDarray((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7), dtype=object)
    >>> conjugate(x)
    CDarray[(x_0, -x_1, -x_2, -x_3, -x_4, -x_5, -x_6, -x_7), dtype=object)]
        
    or
        
    >>> x.C
    CDarray[(x_0, -x_1, -x_2, -x_3, -x_4, -x_5, -x_6, -x_7), dtype=object)]
    """    
    return CDarray(np.r_[arr[:1], -arr[1:len(arr)]], int(np.log2(len(arr))))


def CDproduct(arr1, arr2):
    """
    dim1, dim2 = len(arr1), len(arr2)
    n1, n2 = int(np.log2(dim1)), int(np.log2(dim2))
    if dim1 > dim2:
        arr2 = CDarray(arr2, n1)
        dim = dim1
    elif dim1 < dim2:
        arr1 = CDarray(arr1, n2)
        dim = dim2
    else:
        pass
    """
    dim = len(arr1)
    n = int(np.log2(dim))
    a, b = arr1[:int(dim/2)], arr1[int(dim/2):]
    c, d = arr2[:int(dim/2)], arr2[int(dim/2):]
    if dim == 2:
        x = a[0] * c[0] - d[0] * b[0]
        y = b[0] * c[0] + d[0] * a[0]
        return np.array([S.expand(x), S.expand(y)])
    else:
        x = CDproduct(a, c) - CDproduct(conjugate(d), b)
        y = CDproduct(b, conjugate(c)) + CDproduct(d, a)
        return CDarray(np.r_[x, y], n)


def CDpower(arr, k):
    if not isinstance(arr, CDarray):
        print('%s is not a CDarray-object' % arr)
        return CDarray(np.append([1], [0 for i in range(2**n - len(v))]), int(np.log2(len(v))))
    if isinstance(k, int) and k >= 0:
        if   k == 0:
            return CDarray(1, int(np.log2(len(arr))))
        elif k == 1:
            return arr
        else:
            arr1 = arr
            for i in range(k-1):
                arr = CDproduct(arr, arr1)
            return arr
    else:
        print('Error: %s is NOT 0 or a positive integer' % k)
        return None

    
def simple_form(v):
    s = S.simplify(v)
    if isinstance(s, S.ImmutableDenseNDimArray):
        return CDarray(s, int(np.log2(len(v))))
    else:
        return s

    
def inverse(v):
    return conjugate(v)/N(v)


class CDarray(np.ndarray):
    """
    CDarray(v, n, pure=False)
    
    Return an elemement of the 2^n-dimensional Cayley-Dickson algebra (2^n-nion) as an array-like object.
    
    Parameters
    ----------
    v    : string, integer, float, symbol, list, ndarray or CDarray
    n    : positive ineteger,
        ´n´ of 2^n-dimensions.
    pure : bool
        If ´pure=True´, the CDarray-function returns the pure 2^n-nion.

    Example 1a
    ----------
    >>> x = CDarray('x', 3)
    >>> x
    CDarray([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7], dtype=object)
    
    Example 1b
    ----------    
    >>> x = CDarray('x', 3, pure=True)
    >>> x
    CDarray([0, x_1, x_2, x_3, x_4, x_5, x_6, x_7], dtype=object)

    Example 2a
    ----------
    >>> x = CDarray(3, 4)
    >>> x
    CDarray([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object)

    Example 2b
    ----------
    >>> x = CDarray(3, 4, pure=True)
    >>> x
    CDarray([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=object)

    Example 3a
    ----------
    >>> x = CDarray(numpy.pi, 2)
    >>> x
    CDarray([3.14159265, 0., 0., 0.])
    
    Example 3b
    ----------
    >>> x = CDarray(numpy.pi, 2, pure=True)
    >>> x
    CDarray([0., 3.14159265, 3.14159265, 3.14159265])
    
    Example 4a
    ----------
    >>> x = sympy.Symbol('x')
    >>> x = CDarray('x', 3)
    CDarray([x, 0, 0, 0, 0, 0, 0, 0], dtype=object)
    
    Example 4b
    ----------
    >>> x = sympy.Symbol('x')
    >>> x = CDarray('x', 3, pure=True)
    CDarray([0, x, x, x, x, x, x, x], dtype=object)
  
    etc..
                    
    """
    
    def __new__(cls, v, n, pure=False):
        if isinstance(v, str):
            arr = S.symarray(v, 2**n)
            if pure == True:
                arr[0] = 0
            return np.ndarray.__new__(cls, (2**n,), arr.dtype, buffer=arr)
        elif isinstance(v, (np.ndarray, list, S.ImmutableDenseNDimArray, CDarray)):
            if isinstance(v, (list, S.ImmutableDenseNDimArray)):
                v = np.array(v)
            if len(v) > 2**n:
                n1 = int(np.log2(len(v))) + 1
                print('In this case, the second argument of CDarray must be %s or an integer of more than %s.' % (n1, n1))
                print('Thus %s is NOT transformed to CDarray.' % v)
                return v
            elif len(v) < 2**n:
                if pure == True:
                    arr = np.append(np.append([0], v), [0 for i in range(2**n - len(v) - 1)])
                else:    
                    arr = np.append(v, [0 for i in range(2**n - len(v))])
            else:
                if pure == True:
                    v[0] = 0
                arr = v
        else:
            if pure == True:
                arr = np.append(np.array([0]), [v for i in range(2**n - 1)])
            else:
                arr = np.append(v, [0 for i in range(2**n - 1)])
        return arr.view(cls)

    
    def __init__(self, v, n, pure=False):
        self.n = n
        #self.v = v
        self.e0  = [1] + [ 0 for i in range(2**self.n - 1)]
        #self.eta = [1] + [-1 for i in range(2**self.n - 1)]

        
    def __add__(self, other):
        if isinstance(other, (np.ndarray, list)):
            return np.add(self, other)
        else:            
            return np.add(self, other * np.array(self.e0))
    
    def __radd__(self, other):
        if isinstance(other, (np.ndarray, list)):
            return np.add(other, self)
        else:            
            return np.add(other * np.array(self.e0), self)

        
    def __sub__(self, other):
        if isinstance(other, (np.ndarray, list)):
            return np.subtract(self, other)
        else:            
            return np.subtract(self, other * np.array(self.e0))
    
    def __rsub__(self, other):
        if isinstance(other, (np.ndarray, list)):
            return np.subtract(other, self)
        else:            
            return np.subtract(other * np.array(self.e0), self)

        
    def __eq__(self, other):
        if (type(other)==CDarray) and (len(self)==len(other)):
            for i in range(len(self)):
                ret = False
                if self[i] != other[i]:
                    break
                ret = True
        return ret
 
    
    C = property(conjugate)
    

    def __mul__(self, other):
        if isinstance(other, (int, float, S.Symbol)):
            return np.multiply(self, other)
        else:
            if not isinstance(other, CDarray):
                print('Error: %s is NEITHER scaler NOR CDarray.' % other)
                return None
            if len(self) > len(other):
                arr = CDarray(other, int(np.log2(len(self))))
            #elif len(self) < len(other):
                #return CDproduct(self, other)
            else:
                return CDproduct(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float, S.Symbol)):
            return np.multiply(self, other)
        else:
            return CDproduct(other, self)

        
    def __pow__(self, k):
        if k == -1:
            return inverse(self)
        else:
            return CDpower(self, k)


    simple = property(simple_form)
