
# UDIS-D Warp Training and Testing

It includes scripts for training, testing, and generating warped images. Below are the detailed instructions and functions of each script.

## Training on UDIS-D

To train the model on the UDIS-D dataset, make sure you have set the correct path for the training dataset in `Warp/Codes/train.py`.

### Training Command
After setting the dataset path, run the following command to start training:

```bash
python train.py
```

### Training Script Explanation
The `train.py` script performs the following functions:

#### Dataset Loading:
- Loads training data from the specified path using the TrainDataset class.
- Uses DataLoader to batch and shuffle the data for training.

#### Model Initialization:
- Defines the model using the Network class and moves it to the GPU if available.

#### Optimizer and Scheduler:
- Uses the Adam optimizer with a learning rate of 1e-4 and an ExponentialLR scheduler to decay the learning rate over epochs.

#### Model Checkpoint:
- If a pre-trained model exists, it will be loaded from the checkpoint, along with the optimizer state, and training resumes from the last saved epoch.

#### Loss Calculation:
The script computes two types of losses:
- **Overlap Loss**: Computed using the cal_lp_loss function for overlapping regions of the mesh.
- **Non-overlap Loss**: Computed using a combination of inter_grid_loss and intra_grid_loss for non-overlapping regions.

#### Gradient Clipping:
- Gradient clipping is applied to prevent exploding gradients.

#### Logging with TensorBoard:
- The script logs the training progress, including images and loss statistics, to TensorBoard for visualization.

#### Model Saving:
- The model is saved every 10 epochs or at the final epoch.

## Testing on UDIS-D

### Pre-trained Model
A pre-trained warp model is available for download. You can find the pre-trained model at Google Drive.

### PSNR/SSIM Calculation
To evaluate the performance of the model on the testing dataset, set the testing dataset path in `Warp/Codes/test.py` and run:

```bash
python test.py
```

This script calculates the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) between the original and warped images.

### Generating Warped Images and Corresponding Masks
To generate the warped images and their corresponding masks, set the testing dataset path in `Warp/Codes/test_output.py` and run:

```bash
python inference.py
```

#### Functionality:
- This script performs image warping using the trained model and generates the warped images along with corresponding masks.
- The warped images and masks are saved in the original training/testing dataset path.
- Additionally, the results of average fusion are saved in the current working directory.

### Testing on Other Datasets
When testing on other datasets with different scenes and resolutions, you can apply the iterative warp adaptation to improve alignment performance.

Set the 'path/img1_name/img2_name'.
Modify the 'path/img1_name/img2_name' in `Warp/Codes/test_other.py` (By default, both img1 and img2 are placed under 'path') and run:

```bash
python test_other.py
```

#### Functionality:
- This script applies the iterative warp adaptation process, which improves the alignment of images in different scenes and resolutions.
- The results before and after adaptation will be saved in the specified 'path'.

## Additional Information

### Important Functions in train.py
- **train()**: Main function that controls the entire training process. Sets up the dataset, model, optimizer, scheduler, and handles model saving and loading from checkpoints.

#### Loss Functions:
- **cal_lp_loss()**: Calculates loss for overlapping regions.
- **inter_grid_loss()**: Computes the inter-grid loss for non-overlapping areas.
- **intra_grid_loss()**: Computes intra-grid loss for non-overlapping areas.

### Logging in TensorBoard
The following information is logged to TensorBoard:
- **Images**: Input images, warped images, and meshes are visualized in TensorBoard.
- **Scalars**: Training losses (total, overlap, non-overlap) and learning rate are logged for each iteration.


