import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import cv2
#from torch_homography_model import build_model
from network import get_stitched_result, Network, build_new_ft_model
import glob
from loss import cal_lp_loss2
import torchvision.transforms as T

#import PIL
resize_512 = T.Resize((512, 512))

# Function to load and preprocess two images for training
def loadSingleData(data_path, img1_name, img2_name):
    # load image1
    input1 = cv2.imread(data_path + img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0  # Normalize to [-1, 1]
    input1 = np.transpose(input1, [2, 0, 1])  # Convert to (C, H, W)

    # load image2
    input2 = cv2.imread(data_path + img2_name)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0  # Normalize to [-1, 1]
    input2 = np.transpose(input2, [2, 0, 1])  # Convert to (C, H, W)

    # Convert to tensor and add batch dimension
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)

# Path to the current project directory
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# Directory to save model files
MODEL_DIR = os.path.join(last_path, 'model')

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Training function
def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Initialize the network model
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()  # Move the model to GPU if available

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Load existing model if available
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print('Loaded model from {}!'.format(model_path))
    else:
        start_epoch = 0
        print('Training from scratch!')

    # Load the dataset (only one pair of images in this case)
    input1_tensor, input2_tensor = loadSingleData(data_path=args.path, img1_name=args.img1_name, img2_name=args.img2_name)
    if torch.cuda.is_available():
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    # Resize the input images to 512x512
    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []

    print("################## Start iteration #######################")
    for epoch in range(start_epoch, start_epoch + args.max_iter):
        net.train()

        optimizer.zero_grad()

        # Get the output from the network
        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out['warp_mesh']
        warp_mesh_mask = batch_out['warp_mesh_mask']
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']

        # Calculate the LP loss (local photometric loss)
        total_loss = cal_lp_loss2(input1_tensor_512, warp_mesh, warp_mesh_mask)
        total_loss.backward()

        # Clip the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        current_iter = epoch - start_epoch + 1
        print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))

        loss_list.append(total_loss)

        # Save intermediate results during the first iteration
        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
            cv2.imwrite(args.path + 'before_optimization.jpg', output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + 'before_optimization_mesh.jpg', output['stitched_mesh'])

        # Check for convergence after a few iterations
        if current_iter >= 4:
            if torch.abs(loss_list[current_iter - 4] - loss_list[current_iter - 3]) <= 1e-4 and torch.abs(loss_list[current_iter - 3] - loss_list[current_iter - 2]) <= 1e-4 \
            and torch.abs(loss_list[current_iter - 2] - loss_list[current_iter - 1]) <= 1e-4:
                with torch.no_grad():
                    output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

                path = args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + ".jpg"
                cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
                cv2.imwrite(args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
                cv2.imwrite(args.path + 'warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1, 2, 0))
                cv2.imwrite(args.path + 'warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1, 2, 0))
                cv2.imwrite(args.path + 'mask1.jpg', output['mask1'][0].cpu().detach().numpy().transpose(1, 2, 0))
                cv2.imwrite(args.path + 'mask2.jpg', output['mask2'][0].cpu().detach().numpy().transpose(1, 2, 0))
                break

        # Save results after the final iteration
        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

            path = args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + ".jpg"
            cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
            cv2.imwrite(args.path + 'warp1.jpg', output['warp1'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + 'warp2.jpg', output['warp2'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + 'mask1.jpg', output['mask1'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + 'mask2.jpg', output['mask2'][0].cpu().detach().numpy().transpose(1, 2, 0))

        # Step the scheduler for learning rate adjustment
        scheduler.step()

    print("################## End iteration #######################")

# Main function to parse arguments and start training
if __name__ == "__main__":
    print('<==================== Setting arguments ===================>\n')

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--path', type=str, default='../../Carpark-DHW/')
    parser.add_argument('--img1_name', type=str, default='input1.jpg')
    parser.add_argument('--img2_name', type=str, default='input2.jpg')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Start training
    train(args)
