import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    for i in range(len(x)):
        base_poly = np.poly1d(1.0)
        for j in range(len(x)):
            if i != j:
                base_poly *= np.poly1d([1.0 / (x[i] - x[j]), -x[j] / (x[i] - x[j])])

        base_functions.append(base_poly)
        polynomial += y[i] * base_poly

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    for i in range(len(x)-1):
        # Construct the matrix of coefficients
        A = np.array([
            [x[i]**3, x[i]**2, x[i], 1],           
            [x[i+1]**3, x[i+1]**2,x[i+1],1], 
            [3*x[i]**2, 2*x[i], 1, 0],         
            [3*x[i+1]**2, 2*x[i+1],1,0] 
        ])
        # Construct the vector of function and derivative values
        B = np.array([
            y[i],
            y[i + 1],
            yp[i],
            yp[i + 1]
        ])
        coefficients = np.linalg.solve(A, B)

        # Create a polynomial for the current interval using the coefficients
        poly = np.poly1d(coefficients)

        spline.append(poly)
    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    A = np.zeros((4*len(x)-4, 4*len(x)-4))
    f = np.zeros(4*len(x)-4)

    for i in range(len(x)-2):
        col_A = i*4
        row_A = i*4
        # row A
        A[row_A,col_A] = 1.0
        A[row_A,col_A+1] = x[i]
        A[row_A,col_A+2] = x[i]**2
        A[row_A,col_A+3] = x[i]**3
        f[row_A] = y[i]
        # row A +1
        A[row_A+1,col_A] = 1.0
        A[row_A+1,col_A+1] = x[i+1]
        A[row_A+1,col_A+2] = x[i+1]**2
        A[row_A+1,col_A+3] = x[i+1]**3
        f[row_A+1] = y[i+1]
        # row A +2
        A[row_A+2,col_A] = 0.0
        A[row_A+2,col_A+1] = 1.0
        A[row_A+2,col_A+2] = 2.0*x[i+1]
        A[row_A+2,col_A+3] = 3.0*x[i+1]**2
        A[row_A+2,col_A+4] = 0.0
        A[row_A+2,col_A+5] = -1.0
        A[row_A+2,col_A+6] = -2.0*x[i+1]
        A[row_A+2,col_A+7] = -3.0*x[i+1]**2
        f[row_A+2] = 0.0

        # row A +3
        A[row_A+3,col_A] = 0.0
        A[row_A+3,col_A+1] = 0.0
        A[row_A+3,col_A+2] = 2.0
        A[row_A+3,col_A+3] = 6.0*x[i+1]
        A[row_A+3,col_A+4] = 0.0
        A[row_A+3,col_A+5] = 0.0
        A[row_A+3,col_A+6] = -2.0
        A[row_A+3,col_A+7] = -6.0*x[i+1]
        f[row_A+3] = 0.0
    n = len(x)
    # letzten 2 Zeile ohne randbedingung
    col_A = n*4-8
    row_A = n*4-8
    i = n-2
    # row A
    A[row_A,col_A] = 1.0
    A[row_A,col_A+1] = x[i]
    A[row_A,col_A+2] = x[i]**2
    A[row_A,col_A+3] = x[i]**3
    f[row_A] = y[i]
    # row A +1
    A[row_A+1,col_A] = 1.0
    A[row_A+1,col_A+1] = x[i+1]
    A[row_A+1,col_A+2] = x[i+1]**2
    A[row_A+1,col_A+3] = x[i+1]**3
    f[row_A+1] = y[i+1]
    # 1.Randbedingung
    row_A =n*4-6
    col_A =0
    i = 0
    A[row_A,col_A] = 0.0
    A[row_A,col_A+1] = 0.0
    A[row_A,col_A+2] = 2.0
    A[row_A,col_A+3] = 6*x[i]
    f[row_A] = 0.0
    # 2.Randbedingung
    row_A = n*4-5
    col_A = n*4 -8
    i = n -1
    A[row_A,col_A] = 0.0
    A[row_A,col_A+1] = 0.0
    A[row_A,col_A+2] = 2.0
    A[row_A,col_A+3] = 6*x[i]
    f[row_A] = 0.0
    # TODO solve linear system for the coefficients of the spline
    coefficients = np.linalg.solve(A, f)
    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n - 1):
        a = coefficients[i*4]
        b = coefficients[i*4+1]
        c = coefficients[i*4+2]
        d = coefficients[i*4+3]
        poly = np.poly1d([d, c, b, a])
        spline.append(poly)
    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions
    A = np.zeros((4*len(x)-4, 4*len(x)-4))
    f = np.zeros(4*len(x)-4)

    for i in range(len(x)-2):
        col_A = i*4
        row_A = i*4
        # row A
        A[row_A,col_A] = 1.0
        A[row_A,col_A+1] = x[i]
        A[row_A,col_A+2] = x[i]**2
        A[row_A,col_A+3] = x[i]**3
        f[row_A] = y[i]
        # row A +1
        A[row_A+1,col_A] = 1.0
        A[row_A+1,col_A+1] = x[i+1]
        A[row_A+1,col_A+2] = x[i+1]**2
        A[row_A+1,col_A+3] = x[i+1]**3
        f[row_A+1] = y[i+1]
        # row A +2
        A[row_A+2,col_A] = 0.0
        A[row_A+2,col_A+1] = 1.0
        A[row_A+2,col_A+2] = 2.0*x[i+1]
        A[row_A+2,col_A+3] = 3.0*x[i+1]**2
        A[row_A+2,col_A+4] = 0.0
        A[row_A+2,col_A+5] = -1.0
        A[row_A+2,col_A+6] = -2.0*x[i+1]
        A[row_A+2,col_A+7] = -3.0*x[i+1]**2
        f[row_A+2] = 0.0

        # row A +3
        A[row_A+3,col_A] = 0.0
        A[row_A+3,col_A+1] = 0.0
        A[row_A+3,col_A+2] = 2.0
        A[row_A+3,col_A+3] = 6.0*x[i+1]
        A[row_A+3,col_A+4] = 0.0
        A[row_A+3,col_A+5] = 0.0
        A[row_A+3,col_A+6] = -2.0
        A[row_A+3,col_A+7] = -6.0*x[i+1]
        f[row_A+3] = 0.0
    n = len(x)
    # letzten 2 Zeile ohne randbedingung
    col_A = n*4-8
    row_A = n*4-8
    i = n-2
    # row A
    A[row_A,col_A] = 1.0
    A[row_A,col_A+1] = x[i]
    A[row_A,col_A+2] = x[i]**2
    A[row_A,col_A+3] = x[i]**3
    f[row_A] = y[i]
    # row A +1
    A[row_A+1,col_A] = 1.0
    A[row_A+1,col_A+1] = x[i+1]
    A[row_A+1,col_A+2] = x[i+1]**2
    A[row_A+1,col_A+3] = x[i+1]**3
    f[row_A+1] = y[i+1]
    # 1.Randbedingung
    row_A =n*4-6
    col_A =0
    i = 0
    A[row_A,col_A] = 0.0
    A[row_A,col_A+1] = 1.0
    A[row_A,col_A+2] = 2.0*x[i]
    A[row_A,col_A+3] = 3.0*x[i]**2
    f[row_A] = 0.0
    A[row_A+1,col_A] = 0.0
    A[row_A+1,col_A+1] = 0.0
    A[row_A+1,col_A+2] = 2.0
    A[row_A+1,col_A+3] = 6.0*x[i]
    f[row_A+1] = 0.0
    # 2.Randbedingung
    col_A = n*4 -8
    i = n -1
    A[row_A,col_A] = 0.0
    A[row_A,col_A+1] = -1.0
    A[row_A,col_A+2] = -2.0*x[i]
    A[row_A,col_A+3] = -3.0*x[i]**2
    A[row_A+1,col_A] = 0.0
    A[row_A+1,col_A+1] = 0.0
    A[row_A+1,col_A+2] = -2.0
    A[row_A+1,col_A+3] = -6.0*x[i]
    # TODO solve linear system for the coefficients of the spline
    coefficients = np.linalg.solve(A, f)
    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n - 1):
        a = coefficients[i*4]
        b = coefficients[i*4+1]
        c = coefficients[i*4+2]
        d = coefficients[i*4+3]
        poly = np.poly1d([d, c, b, a])
        spline.append(poly)
    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
