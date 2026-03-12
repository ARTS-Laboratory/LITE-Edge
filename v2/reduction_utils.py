# Copyright (c) UofSC ARTS Lab, 2025 - 2026
# Utilities for creating low-rank approximations of the LSTM models


import numpy as np


# Takes a matrix as input and returns a low-rank approximation of that matrix to
# rank r using the Eckart-Young-Mirsky theorem. r must be less than min(m, n).
def convert_to_rank(matrix: np.ndarray, r: int) -> np.ndarray:
	u, s, v = np.linagl.svd(matrix)

	target_rank = np.linalg.matrix_rank(merged_w) - r

	if target_rank < 0:
		raise Exception('Error: target rank must be less than the rank of the original matrix.')

	reduced_s = s[:target_rank, :target_rank]
	reduced_u = u[:, :target_rank]
	reduced_vt = vt[:target_rank, :]

	return reduced_u @ reduced_s @ reduced_vt


