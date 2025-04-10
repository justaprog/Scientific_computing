import numpy as np

####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # TODO: set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    if ival_size == -1.0:
        ival_size = 10 * np.finfo(np.float64).eps

    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    n_iterations = 0
    # TODO: loop until final interval is found, stop if max iterations are reached
    while n_iterations < n_iters_max and np.abs(rival - lival) > ival_size:
        # Bisection step
        mid_point = (lival + rival) / 2
        fm = f(mid_point)

        # Update interval
        if fm == 0.0:
            root = mid_point
            break
        elif np.sign(fm) == np.sign(fl):
            lival = mid_point
            fl = fm
        else:
            rival = mid_point
            fr = fm

        n_iterations += 1

    # TODO: calculate final approximation to root
    root = (lival + rival) / 2

    return root


def func_f(x):
    return x**3 - 2*x + 2 # -1.76929235423863

def deri_f(x):
    return 3 * x**2 - 2

def func_g(x):
    return 6*x/(x**2 + 1)

def deri_g(x):
    return 6 * (1 - x**2) / (x**2 + 1)**2

def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # TODO: chose meaningful convergence criterion eps, e.g 10 * eps
    eps = 10 * np.finfo(np.float64).eps

    # Initialize iteration
    f_root = f(root)
    df_root = df(root)
    n_iterations = 0

    # TODO: loop until convergence criterion eps is met
    while n_iterations < n_iters_max:
        # TODO: return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if np.abs(df_root) < eps or np.abs(root) > 1e5:
            return root, n_iters_max+1

        # TODO: update root value and function/dfunction values
        root = root - f_root / df_root
        f_root = f(root)
        df_root = df(root)
        # TODO: avoid infinite loops and return (root, n_iters_max+1)
        if np.isnan(root) or np.isinf(root):
            return root, n_iters_max+1

        n_iterations += 1

    return root, n_iterations

####################################################################################################
# Exercise 2: Newton Fractal


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it 
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)
    # TODO: iterate over sampling grid
    for i in range(sampling.shape[0]):
        for j in range(sampling.shape[1]):    
            # TODO: run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            z, n_iterations = find_root_newton(f, df, sampling[i,j], n_iters_max)
  
            # TODO: determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            # index = 0
            index = np.argmin(np.abs(roots - z))

            # TODO: write the index and the number of needed iterations to the result
            result[i, j] = np.array([index, n_iterations])


    return result


####################################################################################################
# Exercise 3: Minimal Surfaces

def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    # initialize area
    area = 0.0

    # TODO: iterate over all triangles and sum up their area
    for triangle in f:
        x0, x1, x2 = v[triangle]
        x0_x1 = x1 - x0
        x0_x2= x2 - x0
        cross_product = np.cross(x0_x1, x0_x2)
        triangle_area = 0.5 * np.linalg.norm(cross_product)
        area += triangle_area

    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)

    # TODO: iterate over all triangles and sum up the vertices gradients
    for triangle in f:
        A, B, C = v[triangle]
        AB = B - A
        AC = C - A
        BC = C - B
        n = np.cross(AB, AC)/np.linalg.norm(np.cross(AB, AC))

        grad_A = 0.5 * np.cross(BC, n)
        grad_B = 0.5 * np.cross(n, AC) 
        grad_C = 0.5 * np.cross(AB, n)

        gradient[triangle[0]] += grad_A
        gradient[triangle[1]] += grad_B
        gradient[triangle[2]] += grad_C

    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float=1e-6, ste: float=1.0, fac: float=0.5) -> (bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """


    # TODO: calculate gradient and area before changing the surface
    gradient = surface_area_gradient(v,f)
    area = surface_area(v, f)

    # TODO: calculate indices of vertices whose position can be changed
    movable_indices = np.setdiff1d(np.arange(len(v)), c)
    # TODO: find suitable step size so that area can be decreased, don't change v yet

        # Initialize updated_v with the current vertex positions
    updated_v = v.copy()

    # Perform line search to find optimal step size
    step = ste
    max_steps = 20

    for _ in range(max_steps):
        updated_v_temp = updated_v.copy()

        for i in movable_indices:
            # Update vertex position in v
            updated_v_temp[i] = updated_v[i] - step * gradient[i]

        # Calculate the new area
        updated_area = surface_area(updated_v_temp, f)

        if updated_area < area:
            # If the new area is smaller, update the vertices and area
            updated_v = updated_v_temp
            area = updated_area

            if abs(area - updated_area) < epsilon:
                return True, area, updated_v, gradient

        # Adjust step size
        step *= fac

    # Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)
    return False, area, updated_v, gradient



if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
