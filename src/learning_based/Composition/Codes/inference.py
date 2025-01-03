# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
from network import build_model, Network
from dataset import *
import os
import numpy as np
import cv2

# Define the directory to save/load model checkpoints
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


def test(args):
    # Set CUDA device configuration
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load the test dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # Initialize the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # Load the latest checkpoint if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')
        return

    # Prepare output directories
    path_learn_mask1 = '../learn_mask1/'
    if not os.path.exists(path_learn_mask1):
        os.makedirs(path_learn_mask1)
    path_learn_mask2 = '../learn_mask2/'
    if not os.path.exists(path_learn_mask2):
        os.makedirs(path_learn_mask2)
    path_final_composition = '../composition/'
    if not os.path.exists(path_final_composition):
        os.makedirs(path_final_composition)

    print("################## Start Testing #######################")
    net.eval()
    for i, batch_value in enumerate(test_loader):
        # Load the batch of test data
        warp1_tensor = batch_value[0].float()
        warp2_tensor = batch_value[1].float()
        mask1_tensor = batch_value[2].float()
        mask2_tensor = batch_value[3].float()

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()

        # Forward pass through the network without computing gradients
        with torch.no_grad():
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

        # Extract outputs from the network
        stitched_image = batch_out['stitched_image']
        learned_mask1 = batch_out['learned_mask1']
        learned_mask2 = batch_out['learned_mask2']

        # Convert outputs to images
        stitched_image = ((stitched_image[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        learned_mask1 = (learned_mask1[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)
        learned_mask2 = (learned_mask2[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)

        # Save outputs to respective directories
        path = path_learn_mask1 + str(i + 1).zfill(6) + ".jpg"
        cv2.imwrite(path, learned_mask1)
        path = path_learn_mask2 + str(i + 1).zfill(6) + ".jpg"
        cv2.imwrite(path, learned_mask2)
        path = path_final_composition + str(i + 1).zfill(6) + ".jpg"
        cv2.imwrite(path, stitched_image)

        # Print progress
        print('i = {}'.format(i + 1))


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')  # Specify which GPU to use
    parser.add_argument('--batch_size', type=int, default=1)  # Batch size for testing

    # Path to the test data
    parser.add_argument('--test_path', type=str, default='../../../../data/UDIS-D/testing/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)

    # Start testing
    test(args)
