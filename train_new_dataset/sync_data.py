# Copyright ARTS Lab, 2024
# CLI tool for synchronizing data from two accelerometers over time.
# This was made as a CLI tool for convienence. If you would like to have
# its functionality in another python script, import the sync_data function.

import numpy as np
from numpy import ndarray
from scipy.signal import find_peaks, correlate
import sys
import os


def main():
    # Small hint in case the user forgets
    if '--help' in sys.argv[1]:
        print("python sync_data.py '<signal1.csv>', '<signal2.csv'>," +
              "'<ouput_directory>'")
    try:
        signal1 = np.genfromtxt(sys.argv[1], delimiter=',')[:, 1]
        signal2 = np.genfromtxt(sys.argv[2], delimiter=',')[:, 1]
    except Exception:
        print("Error: Unable to load files.")
        exit()

    signal1_name = sys.argv[1].split('/')[-1]
    signal2_name = sys.argv[2].split('/')[-1]

    corrected_dataset_1, corrected_dataset_2 = sync_data(signal1, signal2)

    try:
        os.makedirs(sys.argv[3], exist_ok=True)
        np.savetxt(sys.argv[3] + '/' + signal1_name,
                   corrected_dataset_1, delimiter=',')
        np.savetxt(sys.argv[3] + '/' + signal2_name,
                   corrected_dataset_2, delimiter=',')
    except Exception:
        print("Error: Unable to write to output directory.")


# Syncronize two sets of accelerometer data at their first peaks.
def sync_data(signal1: ndarray, signal2: ndarray) -> (ndarray, ndarray):
    correlation = correlate(signal1, signal2)

    max = 0
    for i, value in correlation:
        if value > max:
            max = value
            lag = i

    output1 = np.zeros(signal1.size - lag)
    output2 = np.zeros(signal1.size - lag)

    if max >= 0:  # Signal 1 needs to be shifted left
        for i, _ in enumerate(output1):
            output1[i] = signal1[i + lag]
            output2[i] = signal2[i]
    else:  # Signal 2 needs to be shifted left
        for i, _ in enumerate(output1):
            output1[i] = signal1[i]
            output2[i] = signal2[i + lag]

    return (output1, output2)


if __name__ == '__main__':
    main()
