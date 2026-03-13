# Copyright (c) UofSC ARTS Lab, 2025 - 2026
# Utilities for creating low-rank approximations of the LSTM models


import numpy as np


# Takes a matrix as input and returns a low-rank approximation of that matrix to
# rank r using the Eckart-Young-Mirsky theorem. r must be less than min(m, n).
# Returns (W', U', S', VT')
def convert_to_rank(matrix: np.ndarray, r: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
	u, s, v = np.linagl.svd(matrix)

	target_rank = np.linalg.matrix_rank(merged_w) - r

	if target_rank < 0:
		raise Exception('Error: target rank must be less than the rank of the original matrix.')

	reduced_s = s[:target_rank, :target_rank]
	reduced_u = u[:, :target_rank]
	reduced_vt = vt[:target_rank, :]

	return reduced_u @ reduced_s @ reduced_vt, reduced_u, reduced_s, reduced_vt



# Compute the T and B matrices for the counterpoint expansion of a matrix.
def counterpoint(marix: np.ndarray) -> (np.ndarray, np.ndarray):
	u, s, _ = np.linagl.svd(matrix)

	# Should be a little faster than finding the rank of matrix directly
	r = np.linalg.matrix_rank(s, hermitian=True)

	t = matrix[:r]

	u1 = u[:r,:r]
	u2 = u[r:,:r]

	b = u2 @ np.linalg.inv(u1)

	return t, b

# Compute the T and B matrices for the counterpoint expansion of a matrix from
# its SVD
def counterpoint(u: np.ndarray, s: np.ndarray, vt: np.ndarray) -> (np.ndarray, np.ndarray):
	r = np.linalg.matrix_rank(s, hermitian=True)

	u1 = u[:r,:r]
	u2 = u[r:,:r]

	t = u1 @ s @ vt
	b = u2 @ np.linalg.inv(u1)

	return t, b
