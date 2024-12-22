import torch
import numpy as np
import cv2

# src_p: shape=(bs, 4, 2)  # Source points, where 'bs' is the batch size.
# det_p: shape=(bs, 4, 2)  # Destination points, where 'bs' is the batch size.
#
# This function solves the Direct Linear Transformation (DLT) algorithm to compute 
# the homography matrix that maps the source points to the destination points.
#
# The system of equations being solved is represented as:
# | x1 y1 1  0  0  0  -x1x2  -y1x2 |   | h1 |
# | 0  0  0  x1 y1 1  -x1y2  -y1y2 | = | h4 |
#                                       | h5 |
#                                       | h6 |
# The result of this equation gives the homography matrix H.

def tensor_DLT(src_p, dst_p):
    bs, _, _ = src_p.shape  # Extract batch size (bs)

    ones = torch.ones(bs, 4, 1)  # Create a tensor of ones for homogeneous coordinates
    if torch.cuda.is_available():  # Check if CUDA is available (use GPU if available)
        ones = ones.cuda()  # Move the tensor to GPU if CUDA is available
        
    # Concatenate the ones tensor to the source points for homogeneous coordinates
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)  # Create a tensor of zeros with the same shape as xy1
    if torch.cuda.is_available():  # Move zeros tensor to GPU if necessary
        zeros = zeros.cuda()

    # Create two new tensors to form parts of the matrix M1 for the DLT algorithm
    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)

    # Reshape and concatenate the tensors to form M1, which will be used in the DLT
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)

    # Compute the outer product of the destination and source points and reshape it
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),  # Reshape destination points to fit matrix multiplication
        src_p.reshape(-1, 1, 2),  # Reshape source points to fit matrix multiplication
    ).reshape(bs, -1, 2)

    # Form the matrix A and the vector b for the system of equations Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)  # Reshape destination points as a column vector
    
    # Solve for the homography vector h by computing the inverse of A and multiplying with b
    Ainv = torch.inverse(A)  # Compute the inverse of matrix A
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)  # Compute the homography vector h (flattened)

    # Reshape h8 to form the homography matrix H (3x3 matrix)
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(bs, 3, 3)
    
    return H  # Return the homography matrix
