
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    (n,m) = A.shape

    if (n != m or b.shape != (m,)):
        raise ValueError("matrix and vector sizes are incompatible ")
    # TODO: Perform gaussian elimination
    for k in range(m):
        if A[k,k] == 0.0:
            if use_pivoting == False:
                raise ValueError("privoting is necessary")
            else: 
                pivot = 0
                pivot_zeile = k
                for p in range(k,m):
                    if(abs(A[p,k]) >= pivot):
                        pivot = abs(A[p,k])
                        pivot_zeile = p     
                A[[pivot_zeile,k]] = A[[k,pivot_zeile]]
                b[[pivot_zeile,k]] = b[[k,pivot_zeile]]
        for i in range(k+1, m):
            m_ik = -1*(A[i,k]/A[k,k])
            b[i] = b[i] + m_ik* b[k]
            for j in range(m):
                A[i,j] = A[i,j] + m_ik* A[k,j]
    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    (n,m) = A.shape

    if (n != m or b.shape != (m,)):
        raise ValueError("matrix and vector sizes are incompatible ")
    # TODO: Initialize solution vector with proper size
    x = np.zeros(m)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    if A[m-1,m-1] == 0:
        raise ValueError("no solution")
    x[m-1] = b[m-1]/A[m-1,m-1]
    for i in range(m-2,-1,-1):
        if(A[i][i] == 0):
            raise ValueError("no solutions")
        x[i] = b[i]
        for j in range(i+1,m):
            x[i] = x[i] - A[i,j]*x[j]
        x[i] = x[i]/A[i,i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L : Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError("no symmetry")
    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    L[0,0] = np.sqrt(M[0,0])
    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i,i] = M[i,i]
                sum_di = 0
                for k in range(i):
                    sum_di = sum_di + L[i,k]*L[i,k]
                if M[i,i] - sum_di < 0:
                    raise ValueError("non postive definite matrix")
                L[i,i] = np.sqrt(M[i,i] - sum_di)
            else:
                sum_else = 0
                for k in range(j):
                    sum_else = sum_else + L[i,k]*L[j,k]
                L[i,j] = (M[i,j] - sum_else)/L[j,j]   
    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if (n != m or b.shape != (m,)):
        raise ValueError("not quadrat matrix or b and L dont match")
    for i in range(m):
        for j in range(i + 1,m):
            if(L[i][j] != 0): 
                raise ValueError("bot lower triangular matrix")
    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)
    y = np.zeros(m)

    y[0] = b[0]/L[0,0]
    for i in range(m):
        if(L[i][i] == 0):
            raise ValueError("no solutions")
        y[i] = b[i]
        for j in range(i):
            y[i] = y[i] - L[i,j]*y[j]
        y[i] = y[i]/L[i,i]
    x = back_substitution(L.transpose(),y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_rays*n_shots, n_grid*n_grid))
    # TODO: Initialize intensity vector
    g = np.zeros(n_rays*n_shots)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    for i in range (n_shots):
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        for j in range(n_rays):
            g[j+i*n_rays] = intensities[j]
            for in_s in range(len(ray_indices)):
                L[ray_indices[in_s] + i*n_rays, isect_indices[in_s]] = lengths[in_s]
        theta = theta + 3.14/n_shots

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)
    M = np.dot(np.transpose(L),L)
    g = np.dot(np.transpose(L),g)
    L = compute_cholesky(M)
    x = solve_cholesky(L,g)
    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            tim[i][j] = x[n_grid*(i)+j]

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    A = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            A[i][j] = 4*(4-i-1)+j
    print(A)