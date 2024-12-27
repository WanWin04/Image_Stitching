import torch
import numpy as np

# Function for transforming an image (U) from target (control points) to source (control points)
# All points should be normalized from -1 ~1

# Compared with torch_tps_transform.py, this version moves some operations from GPU to CPU to save GPU memory

def transformer(U, source, target, out_size):

    # Helper function to repeat a tensor for a specified number of times
    def _repeat(x, n_repeats):
        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        # Repeat tensor by multiplying and reshaping
        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    # Interpolation function for sampling pixels from the image based on transformed coordinates
    def _interpolate(im, x, y, out_size):

        num_batch, num_channels , height, width = im.size()

        # Image dimensions
        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        # Boundaries for x and y coordinates
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # Scale x, y coordinates to the image size
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # Sampling logic: find nearest neighbor coordinates
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # Clamp to ensure we don't go out of bounds
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        # Prepare necessary dimensions for indexing
        dim2 = torch.from_numpy(np.array(width))
        dim1 = torch.from_numpy(np.array(width * height))

        base = _repeat(torch.arange(0,num_batch) * dim1, out_height * out_width)

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()

        # Calculate indices for bilinear interpolation
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Reorganize image channels
        im = im.permute(0,2,3,1)
        im_flat = im.reshape([-1, num_channels]).float()

        # Gather pixel values for each index
        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch,num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # Convert to float for computation
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        # Compute interpolation weights
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        # Combine interpolated values to get final output
        output = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return output

    # Generate a meshgrid based on the source points to transform the image
    def _meshgrid(height, width, source):

        source = source.cpu()

        # Generate grid of coordinates between -1 and 1
        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))

        # Flatten the grid
        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        # Calculate distances from source points to grid points
        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:,:,0], 2)
        py = torch.unsqueeze(source[:,:,1], 2)

        # Calculate the distance matrix and apply a logarithmic transformation
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)

        # Prepare grid for transformation
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)
        ones = torch.ones_like(x_t_flat_g)

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)

        # Move grid to GPU if available
        if torch.cuda.is_available():
            grid = grid.cuda()

        return grid

    # Apply transformation matrix to the source points and the input image
    def _transform(T, source, input_dim, out_size):
        num_batch, num_channels, height, width = input_dim.size()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source)

        # Transform image based on the transformation matrix
        T_g = torch.matmul(T, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]
        x_s_flat = x_s.reshape([-1])
        y_s_flat = y_s.reshape([-1])

        # Interpolate the transformed image
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        output = output.permute(0,3,1,2)

        return output

    # Solve the system of equations to compute the transformation matrix T
    def _solve_system(source, target):
        num_batch  = source.size()[0]
        num_point  = source.size()[1]

        # Create ones for the source points
        ones = torch.ones(num_batch, num_point, 1).float()
        if torch.cuda.is_available():
            ones = ones.cuda()

        # Augment source points
        p = torch.cat([ones, source], 2)

        # Compute pairwise distance matrix and apply logarithmic transformation
        p_1 = p.reshape([num_batch, -1, 1, 3])
        p_2 = p.reshape([num_batch, 1, -1, 3])
        d2 = torch.sum(torch.square(p_1 - p_2), 3)

        r = d2 * torch.log(d2 + 1e-6)

        # Create system matrix W
        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        W_0 = torch.cat((p, r), 2)
        W_1 = torch.cat((zeros, p.permute(0,2,1)), 2)
        W = torch.cat((W_0, W_1), 1)

        # Compute the inverse of W
        W_inv = torch.inverse(W.type(torch.float64))

        # Augment target points
        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()

        tp = torch.cat((target, zeros2), 1)

        # Compute the transformation matrix T
        T = torch.matmul(W_inv, tp.type(torch.float64))
        T = T.permute(0, 2, 1)

        return T.type(torch.float32)

    # Compute transformation matrix T
    T = _solve_system(source, target)

    # Apply transformation to input image
    output = _transform(T, source, U, out_size)

    return output  # Return the transformed image
