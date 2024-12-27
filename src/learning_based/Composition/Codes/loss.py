import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to compute L-norm loss (default is L1 loss)
def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

# Function to extract boundary of the given mask
def boundary_extraction(mask):
    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)

    # Define kernel for dilation
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel, in_channel, 3, 3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    # Perform dilation and extract boundary
    x = F.conv2d(1 - mask, weight, stride=1, padding=1)
    x = torch.where(x < 1, zeros, ones)
    for _ in range(7):  # Repeat dilation steps
        x = F.conv2d(x, weight, stride=1, padding=1)
        x = torch.where(x < 1, zeros, ones)

    return x * mask  # Multiply by the original mask to keep the boundary region

# Function to calculate boundary term loss
def cal_boundary_term(inpu1_tensor, inpu2_tensor, mask1_tensor, mask2_tensor, stitched_image):
    boundary_mask1 = mask1_tensor * boundary_extraction(mask2_tensor)
    boundary_mask2 = mask2_tensor * boundary_extraction(mask1_tensor)

    loss1 = l_num_loss(inpu1_tensor * boundary_mask1, stitched_image * boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tensor * boundary_mask2, stitched_image * boundary_mask2, 1)

    return loss1 + loss2, boundary_mask1

# Function to calculate smoothness term for stitched images
def cal_smooth_term_stitch(stitched_image, learned_mask1):
    delta = 1
    # Compute horizontal and vertical differences for masks and images
    dh_mask = torch.abs(learned_mask1[:, :, 0:-delta, :] - learned_mask1[:, :, delta:, :])
    dw_mask = torch.abs(learned_mask1[:, :, :, 0:-delta] - learned_mask1[:, :, :, delta:])
    dh_diff_img = torch.abs(stitched_image[:, :, 0:-delta, :] - stitched_image[:, :, delta:, :])
    dw_diff_img = torch.abs(stitched_image[:, :, :, 0:-delta] - stitched_image[:, :, :, delta:])

    # Compute pixel-wise loss
    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

# Function to calculate smoothness term for difference between images
def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):
    # Compute squared difference feature within the overlap region
    diff_feature = torch.abs(img1 - img2)**2 * overlap

    delta = 1
    # Compute horizontal and vertical differences for masks and difference feature
    dh_mask = torch.abs(learned_mask1[:, :, 0:-delta, :] - learned_mask1[:, :, delta:, :])
    dw_mask = torch.abs(learned_mask1[:, :, :, 0:-delta] - learned_mask1[:, :, :, delta:])
    dh_diff_img = torch.abs(diff_feature[:, :, 0:-delta, :] + diff_feature[:, :, delta:, :])
    dw_diff_img = torch.abs(diff_feature[:, :, :, 0:-delta] + diff_feature[:, :, :, delta:])

    # Compute pixel-wise loss
    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss
