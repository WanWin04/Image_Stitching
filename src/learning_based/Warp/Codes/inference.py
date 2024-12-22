# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_output_model, Network
from dataset import *
import os
import cv2

# Grid resolution imported from grid_res module
import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

# Set the model directory to store or load model files
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


# Function to draw a mesh on the warped image using local feature points (f_local)
def draw_mesh_on_warp(warp, f_local):
    point_color = (0, 255, 0) # BGR (Green color for the grid lines)
    thickness = 2  # Line thickness for grid lines
    lineType = 8  # Line type for drawing lines

    # Loop through the grid resolution and draw lines connecting local feature points
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            # Skip the last point to avoid index out of bounds
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                # Draw vertical lines for the grid
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                # Draw horizontal lines for the grid
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else:
                # Draw both horizontal and vertical grid lines
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp


# Function to create and save a GIF from a list of image paths
def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)  # Append each image to frames
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)  # Save frames as GIF with 0.5 second per frame
    return


# The main test function for evaluating the network
def test(args):
    # Set CUDA environment to use specific GPU device
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Initialize dataset and data loader for the test set
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # Define the network model
    net = Network()  # Initialize the network
    if torch.cuda.is_available():
        net = net.cuda()  # Move the model to GPU if available

    # Load pre-trained model if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")  # Search for .pth files in the model directory
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)  # Load the checkpoint
        net.load_state_dict(checkpoint['model'])  # Load model weights
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')

    # Create directories for saving output images if they don't exist
    path_ave_fusion = '../ave_fusion/'
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)
    path_warp1 = args.test_path + 'warp1/'
    if not os.path.exists(path_warp1):
        os.makedirs(path_warp1)
    path_warp2 = args.test_path + 'warp2/'
    if not os.path.exists(path_warp2):
        os.makedirs(path_warp2)
    path_mask1 = args.test_path + 'mask1/'
    if not os.path.exists(path_mask1):
        os.makedirs(path_mask1)
    path_mask2 = args.test_path + 'mask2/'
    if not os.path.exists(path_mask2):
        os.makedirs(path_mask2)

    # Set model to evaluation mode
    net.eval()
    for i, batch_value in enumerate(test_loader):
        inpu1_tesnor = batch_value[0].float()  # Input image 1
        inpu2_tesnor = batch_value[1].float()  # Input image 2

        # Move the input tensors to GPU if available
        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        # No gradient calculation during inference
        with torch.no_grad():
            batch_out = build_output_model(net, inpu1_tesnor, inpu2_tesnor)

        # Extract results from the model output
        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']

        # Process and save the results as images
        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        final_mesh2 = final_mesh2[0].cpu().detach().numpy()

        # Save the processed output images
        path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1)
        path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2)
        path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1_mask*255)
        path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2_mask*255)

        # Generate average fusion image
        ave_fusion = final_warp1 * (final_warp1 / (final_warp1 + final_warp2 + 1e-6)) + final_warp2 * (final_warp2 / (final_warp1 + final_warp2 + 1e-6))
        path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, ave_fusion)

        print('i = {}'.format(i+1))

        # Clear GPU memory after each iteration
        torch.cuda.empty_cache()

    print("##################end testing#######################")


# Main entry point for the script
if __name__=="__main__":

    # Argument parser to accept command-line arguments
    parser = argparse.ArgumentParser()

    # GPU and batch size arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Path to test dataset
    parser.add_argument('--test_path', type=str, default='../../../../data/UDIS-D/testing/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
