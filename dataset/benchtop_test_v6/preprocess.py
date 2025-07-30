import numpy as np
import scipy.signal as signal
from numpy import ndarray
import matplotlib.pyplot as plt

def main():
    # Process test 1
    input = np.genfromtxt('./Test 1/input.csv')
    output = np.genfromtxt('./Test 1/output.csv')

    input, output = normalize_data(input, output)

    input, output = align_data(input, output)

    x = range(input.size)
    plt.plot(x, input)
    plt.plot(x, output)
    plt.show()

    np.save('./Test 1/reference.npy', input)
    np.save('./Test 1/package.npy', output)

    # Process test 2
    input = np.genfromtxt('./Test 2/input.csv')
    output = np.genfromtxt('./Test 2/output.csv')

    input, output = normalize_data(input, output)

    input, output = align_data(input, output)

    x = range(input.size)
    plt.plot(x, input)
    plt.plot(x, output)
    plt.show()

    np.save('./Test 2/reference.npy', input)
    np.save('./Test 2/package.npy', output)

    input = np.genfromtxt('./Test 3/input.csv')
    output = np.genfromtxt('./Test 3/output.csv')

    input, output = normalize_data(input, output)

    input, output = align_data(input, output)

    x = range(input.size)
    plt.plot(x, input)
    plt.plot(x, output)
    plt.show()

    np.save('./Test 3/reference.npy', input)
    np.save('./Test 3/package.npy', output)
    

# Aligns the reference and package signals using cross correlation
def align_data(array1: ndarray, array2: ndarray):
    correlation = signal.correlate(array1, array2, mode='full')
    offset = np.argmax(correlation) - array1.size + 1
    print(offset)

    if offset > 0:
        return array1[offset:], array2[:len(array2) - offset]
    else:
        return array1[:len(array1) + offset], array2[offset * -1:]


# Normalize the datasets to 0
def normalize_data(reference: ndarray, package: ndarray):

    # The reference accelerometer is upside-down on the benchtop, so it needs 
    # to be flipped
    return (reference - np.mean(reference)) * -1, package - np.mean(package)


if __name__ == '__main__':
    main()
