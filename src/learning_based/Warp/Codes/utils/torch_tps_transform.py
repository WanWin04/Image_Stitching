import torch
import numpy as np

# transforming an image (U) from target (control points) to source (control points)
# all the points should be normalized from -1 ~ 1

def transformer(U, source, target, out_size):

    def _repeat(x, n_repeats):
        # Repeat tensor `x` for `n_repeats` times, helpful in reshaping indices later.
        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()
        x = torch.matmul(x.reshape([-1, 1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size):
        # Perform bilinear interpolation on the image for the transformed grid
        num_batch, num_channels, height, width = im.size()
        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]
        
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # Normalize (x, y) coordinates back to original image dimensions
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # Compute the four nearest neighbors for bilinear interpolation
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # Ensure indices are within bounds
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy(np.array(width))
        dim1 = torch.from_numpy(np.array(width * height))

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Flatten the image to simplify gathering pixel values
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        # Gather pixel values for each of the four neighbors
        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
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

        # Compute interpolation weights
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        # Weighted sum of the four neighbors
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width, source):
        # Create a grid of coordinates (x, y) normalized to [-1, 1]
        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))

        if torch.cuda.is_available():
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        # Calculate distances between points in meshgrid and control points
        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:, :, 0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:, :, 1], 2)  # [bn, pn, 1]
        
        if torch.cuda.is_available():
            px = px.cuda()
            py = py.cuda()
            
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)  # Radial basis function (RBF) transformation
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        if torch.cuda.is_available():
            ones = ones.cuda()

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, source, input_dim, out_size):
        # Apply the transformation matrix T to the source points
        num_batch, num_channels, height, width = input_dim.size()
        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source)  # [bn, 3+pn, h*w]

        # Apply transformation to grid and map coordinates to source image
        T_g = torch.matmul(T, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        x_s_flat = x_s.reshape([-1])
        y_s_flat = y_s.reshape([-1])

        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        output = output.permute(0, 3, 1, 2)
        return output

    def _solve_system(source, target):
        # Solve for the transformation matrix T using a system of equations
        num_batch = source.size()[0]
        num_point = source.size()[1]

        ones = torch.ones(num_batch, num_point, 1).float()
        if torch.cuda.is_available():
            ones = ones.cuda()
        p = torch.cat([ones, source], 2)  # [bn, pn, 3]

        # Compute pairwise distances between points
        p_1 = p.reshape([num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        # Construct the system matrix W and solve for T
        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]
        W_inv = torch.inverse(W.type(torch.float64))  # Inverse of W

        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()

        tp = torch.cat((target, zeros2), 1)  # [bn, pn+3, 2]
        T = torch.matmul(W_inv, tp.type(torch.float64))  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]
        return T.type(torch.float32)

    # Compute transformation matrix T
    T = _solve_system(source, target)

    # Apply transformation to input image U
    output = _transform(T, source, U, out_size)

    return output
