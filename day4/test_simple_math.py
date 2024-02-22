import simple_math

def test_simple_math():
    # Addition
    assert simple_math.simple_add(1,2) == 3
    assert simple_math.simple_add(99,1) == 100
    assert simple_math.simple_add(-3,-4) == -7
    # Subtraction
    assert simple_math.simple_sub(1,4) == -3
    assert simple_math.simple_sub(4,1) == 3
    assert simple_math.simple_sub(100,1) == 99
    # Multiplication
    assert simple_math.simple_mult(3,4) == 12
    assert simple_math.simple_mult(-3,4) == -12
    assert simple_math.simple_mult(-10,-10) == 100
    # Division
    assert simple_math.simple_div(12,4) == 3
    assert simple_math.simple_div(100,10) == 10
    assert simple_math.simple_div(-100,10) == -10
    # First order polynomial
    assert simple_math.poly_first(2,1,3) == 7
    assert simple_math.poly_first(10,-10,0.5) == -5
    assert simple_math.poly_first(10,100,10) == 200
    # Second order polynomial 
    assert simple_math.poly_second(2,1,2,3) == 17 
    assert simple_math.poly_second(3,-100,10,3) == -43
    assert simple_math.poly_second(2,3,4,5) == 31