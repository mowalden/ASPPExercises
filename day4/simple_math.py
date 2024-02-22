"""
A collection of simple math operations
"""

def simple_add(a,b):
    """
    The sum of two numbers.

    Adds two numbers together, following the arithmetic expression a + b.

    Parameters
    ----------
    a : int, float
        First value to be added to the second value.
    b : int, float
        Second value to be added to the third.

    Returns
    -------
    int, float
        Returns the result of a + b.

    Examples
    --------
    >>> simple_add(1,2)
    3
    >>> simple_add(-2,1)
    -1
    """
    return a+b

def simple_sub(a,b):
    """
    The subtraction of two numbers.

    Subtracts one number from the other, following the arithmetic expression a - b.

    Parameters
    ----------
    a : int, float
        Subtrahend.
    b : int, float
        Minuend.

    Returns
    -------
    int, float
        Returns the result of a - b.

    Examples
    --------
    >>> simple_sub(3,2)
    1
    >>> simple_sub(4,10)
    -6
    """
    return a-b

def simple_mult(a,b):
    """
    The multiplication of two numbers.

    Multiplies two numbers, following the arithmetic expression a * b.

    Parameters
    ----------
    a : int, float
        First number to be multiplied with the second.
    b : int, float
        Second number to be multiplied with the first.

    Returns
    -------
    int, float
        Returns the result of a * b.

    Examples
    --------
    >>> simple_mult(3,2)
    6
    >>> simple_mult(-3,10)
    -30
    """
    return a*b

def simple_div(a,b):
    """
    The division of two numbers.

    Divides one number by the other, following the arithmetic expression a / b.

    Parameters
    ----------
    a : int, float
        Enumerator.
    b : int, float
        Denominator.

    Returns
    -------
    float
        Returns the result of a / b.

    Examples
    --------
    >>> simple_div(3,2)
    6
     >>> simple_div(5,2)
    2.5
    """
    return a/b

def poly_first(x, a0, a1):
    """
    Polynomial of first order.

    Calculates a polynomial of first order, following the arithmetic expression a0 + a1 * x.

    Parameters
    ----------
    x : int, float
        Variable of the polynomial.
    a0 : int, float
        Zeroth order parameter.
    a1 : int, float
        First order parameter.

    Returns
    -------
    int, float
        Returns the result of a0 + a1 * x.

    Examples
    --------
    >>> poly_first(2,1,3)
    7
    >>> poly_first(10,-10,0.5)
    -5
    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """
    Polynomial of second order.

    Calculates a polynomial of second order, following the arithmetic
    expression a0 + a1 * x + a2 * x**2.

    Parameters
    ----------
    x : int, float
        Variable of the polynomial.
    a0 : int, float
        Zeroth order parameter.
    a1 : int, float
        First order parameter.
    a2 : int, float
        Second order parameter.

    Returns
    -------
    int, float
        Returns the result of a0 + a1 * x + a2 * x**2.

    Examples
    --------
    >>> poly_second(2,1,2,3)
    17
    >>> poly_second(3,-100,10,3)
    -43
    """
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
