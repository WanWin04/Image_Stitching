import torch
import numpy as np


def transformer(U, theta, out_size, **kwargs):
    # This function applies a geometric transformation (e.g., affine, projective) to an input tensor using the transformation matrix 'theta'.
    # 'U' represents the input tensor, 'theta' is the transformation matrix, and 'out_size' is the size of the output image after transformation.

    def _repeat(x, n_repeats):
        # Helper function to repeat a tensor 'x' 'n_repeats' times along a specific dimension.
        rep = torch.ones([n_repeats, ]).unsqueeze(0)  # Create a tensor of ones for repeating.
        rep = rep.int()
        x = x.int()
        x = torch.matmul(x.reshape([-1, 1]), rep)  # Perform the repetition by matrix multiplication.
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size):
        # This function performs bilinear interpolation to sample pixel values at transformed coordinates (x, y).
        num_batch, num_channels, height, width = im.size()  # Get the size of the input image tensor.
        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]  # Get output image dimensions.

        # Initialize boundary conditions
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # Normalize the coordinates from [-1, 1] to [0, width-1] and [0, height-1]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # Perform bilinear interpolation by calculating the surrounding pixels
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # Clamp the values to ensure they stay within image bounds
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        
        # Create base indices for batch processing
        dim2 = torch.from_numpy(np.array(width))
        dim1 = torch.from_numpy(np.array(width * height))

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)
        
        if torch.cuda.is_available():
            # Move the tensors to GPU if CUDA is available.
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        
        # Calculate the pixel indices to gather pixel values
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Flatten the image tensor and gather pixel values using the indices
        im = im.permute(0, 2, 3, 1)  # Change the order of dimensions to [batch, height, width, channels]
        im_flat = im.reshape([-1, num_channels]).float()

        # Gather values at each of the four surrounding pixels for bilinear interpolation
        idx_a = idx_a.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)
        
        idx_b = idx_b.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)
        
        idx_c = idx_c.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)
        
        idx_d = idx_d.unsqueeze(-1).long().expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # Calculate the interpolation weights for each surrounding pixel
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        
        # Compute the final output using bilinear interpolation
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width):
        # This function creates a meshgrid for the output image dimensions.
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))

        # Flatten the grid and add a ones row for homogeneous coordinates
        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(theta, input_dim, out_size):
        # This function performs the affine transformation on the input image using the transformation matrix 'theta'.
        num_batch, num_channels, height, width = input_dim.size()

        # Reshape the transformation matrix to 3x3 format
        theta = theta.reshape([-1, 3, 3]).float()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width)  # Create the grid for the output image
        grid = grid.unsqueeze(0).reshape([1, -1])
        
        shape = grid.size()
        grid = grid.expand(num_batch, shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        # Apply the transformation to the grid
        T_g = torch.matmul(theta, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        t_s = T_g[:, 2, :]

        t_s_flat = t_s.reshape([-1])
        
        # Adjust for numerical stability
        small = 1e-7
        smallers = 1e-6 * (1.0 - torch.ge(torch.abs(t_s_flat), small).float())
        t_s_flat = t_s_flat + smallers

        # Normalize the transformed coordinates
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        # Perform the interpolation to get the transformed output
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        output = output.permute(0, 3, 1, 2)  # Change the order of dimensions back to [batch, channels, height, width]
        return output

    # Apply the transformation to the input tensor and return the result
    output = _transform(theta, U, out_size)
    return output
