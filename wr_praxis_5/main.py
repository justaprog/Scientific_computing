import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # TODO: create principal term for DFT matrix
    principal_term = np.exp(-2j* np.pi/ n)
    # TODO: fill matrix with values
    for j in range(n):
        for k in range(n):
            F[j, k] = principal_term**(j*k)
    # TODO: normalize dft matrix
    F = F/np.sqrt(n)

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # TODO: check that F is unitary, if not return false
    # Calculate the conjugate transpose
    conjug = np.conj(matrix.T)
    product = np.dot(conjug, matrix)
    unitary = np.allclose(product, np.identity(matrix.shape[0]))

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals  and extract harmonics out of DFT matrix
    F = dft_matrix(n)
    # Create delta impulse signals and perform Fourier transform
    for k in range(n):
        delta_impulse = np.zeros(n)
        delta_impulse[k] = 1.0  # Delta impulse at position k
        sigs.append(delta_impulse)
        
        # Perform Fourier transform using DFT matrix
        fsig = np.dot(F, delta_impulse)
        fsigs.append(fsig)

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    n = len(data)
    binary_len = len(bin(n))-3  # Calculate the length of binary representation
    new_data = np.zeros(n, dtype='complex128')
    # Perform bit reversal

    for i in range(n):
        binary_str = format(i, f'0{binary_len}b')[::-1]
        swapped_num = int(binary_str,2)
        new_data[swapped_num] = data[i]
    
    return new_data

def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    
    # Get the size of the input data
    n = fdata.size

    # Check if the input length is a power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError("Input length must be a power of 2 for FFT.")

    # TODO: First step of FFT: Shuffle data in bit-reversed order
    new_data = shuffle_bit_reversed_order(data)
    print(new_data)
    fdata = np.asarray(new_data, dtype='complex128')

    # Second step, recursively merge transforms
    for m in range(int(np.log2(n))):  # Iterate over stages
        size = 2 ** (m)
        half_size = 2**(int(np.log2(n))-m-1)
        for i in range(size):  
            for k in range(half_size): 
                u = fdata[i + 2*k*(2**m)]
                fdata[i + 2*k*(2**m)] = u + np.exp(-2j* i*np.pi/ (2**(m+1)))* fdata[i + (2*k+1)*(2**m)]
                fdata[i + (2*k+1)*(2**m)] = u - np.exp(-2j* i*np.pi/ (2**(m+1)))* fdata[i + (2*k+1)*(2**m)]
                

    # Normalize fft signal


    return fdata    


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0
    data = np.zeros(num_samples)
    # TODO: Generate sine wave with proper frequency
    t = np.linspace(0.0, 1.0, num_samples, endpoint=True)
    
    # Generate sine wave with the specified frequency
    data = np.sin(2 * np.pi *f *t)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data
    fourier = np.fft.fft(adata)
    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    cutoff_index = int(bandlimit*len(fourier)/sampling_rate)
    fourier[cutoff_index+1:-cutoff_index] = 0

    # TODO: compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])
    adata_filtered = np.fft.ifft(fourier).real

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    print(fft([-1/2 , 0, 1, 1/4 , -1/2 , 0, 0, 1/4]))

