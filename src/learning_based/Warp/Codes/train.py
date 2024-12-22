import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss


# Get the current path and define directories for saving models and TensorBoard summaries
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
SUMMARY_DIR = os.path.join(last_path, 'summary')  # directory for saving TensorBoard summaries
writer = SummaryWriter(log_dir=SUMMARY_DIR)
MODEL_DIR = os.path.join(last_path, 'model')  # directory for saving models

# Create directories if they don't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def train(args):
    # Set the GPU device for training
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Define dataset and data loader for training
    train_data = TrainDataset(data_path=args.train_path)  # Custom dataset for training
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Initialize the model
    net = Network()  # Define the neural network model
    if torch.cuda.is_available():
        net = net.cuda()  # Move the model to GPU if available

    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)  # Reduce learning rate over time

    # Load the existing model and optimizer state if available
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")  # Get list of saved checkpoints
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)  # Load the latest checkpoint
        net.load_state_dict(checkpoint['model'])  # Load the model weights
        optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch  # Set the scheduler to resume from the correct epoch
        print('Loaded model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('Training from scratch!')


    print("################## Start training #######################")
    score_print_fre = 300  # Frequency of printing training statistics

    # Training loop for each epoch
    for epoch in range(start_epoch, args.max_epoch):

        print("Start epoch {}".format(epoch))
        net.train()  # Set the model to training mode
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        # Iterate through each batch in the training dataset
        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()  # First input image
            inpu2_tesnor = batch_value[1].float()  # Second input image

            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()

            # Zero the gradients before backward pass
            optimizer.zero_grad()

            # Perform forward pass through the model
            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
            # Extract model outputs (warped meshes, transformations, etc.)
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']

            # Compute loss for overlapping regions
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            # Compute loss for non-overlapping regions
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)

            total_loss = overlap_loss + nonoverlap_loss  # Total loss is the sum of both losses
            total_loss.backward()  # Backpropagate the gradients

            # Clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()  # Update model parameters

            # Accumulate losses for logging
            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            print(glob_iter)

            # Log loss and images to TensorBoard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # Log images and metrics to TensorBoard
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1


        # Step the scheduler to adjust learning rate
        scheduler.step()
        
        # Save model checkpoint every 10 epochs or at the final epoch
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)  # Save the model checkpoint

    print("################## End training #######################")


if __name__=="__main__":

    # Set up argument parser for command line inputs
    print('<==================== Setting arguments ===================>\n')
    parser = argparse.ArgumentParser()

    # Add arguments for GPU, batch size, max epoch, and training data path
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='../../../../data/UDIS-D/training/')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Start training
    train(args)
