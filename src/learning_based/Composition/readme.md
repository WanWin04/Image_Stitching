
### Train on UDIS-D

Before starting the training process, you need to create the warped images and corresponding masks during the warp stage.

#### Preparing the Training Data:
- Make sure the images and masks are stored in the correct path. This path is set in `Composition/Codes/train.py`.

#### Training:
Run the following command to start the training process:
```bash
python train.py
```

#### Function of each file related to training:
- `train.py`: Contains the core logic of the training process, including loading data, defining the network, calculating loss, and updating weights.
- `network.py`: Defines the network structure (model) used for the task.
- `dataset.py`: Processes input data and creates batches for training.
- `loss.py`: Defines the loss functions, including:
  - `cal_boundary_term`: Calculates the loss related to the boundary of the stitched image.
  - `cal_smooth_term_stitch`: Calculates the smoothness loss for the stitched image.
  - `cal_smooth_term_diff`: Calculates the loss for differences between images.

---

### Test on UDIS-D

#### Download Pre-trained Model:
You can download the pre-trained warp model from: [Google Drive](https://drive.google.com/file/d/1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30/view)

#### Set Up the Test Data Path:
- Set the path for the test dataset in `Composition/Codes/test.py`.

#### Run the Test:
```bash
python inference.py
```
Component masks and the final results on the UDIS-D dataset will be generated and saved in the current directory.

#### Function of each file related to testing:
- `inference.py`: Runs the test on the UDIS-D dataset to produce the final results.
- `test.py`: Contains the test logic, including loading the pre-trained model and applying it to the test images.

---

### Test on Other Datasets

#### Set Up the Test Data Path:
- Set the `path/` in `Composition/Codes/test_other.py`.

#### Run the Test:
```bash
python test_other.py
```
The results will be generated and saved in the specified `path` directory.

#### Function of the related file:
- `test_other.py`: Runs the test on datasets other than UDIS-D, producing results and saving them in the specified path.

---

### Notes:
- **TensorBoard**: The training process is logged with TensorBoard to visualize metrics such as loss and intermediate images. This data is saved in the `summary/` folder.
- **Checkpoint**: The model is saved periodically after every 10 epochs or upon completion of training in the `model/` folder.
