## Dataset (UDIS-D)
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for more details about this dataset.


## Code
#### Requirement
* numpy
* pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
* scikit-image 
* tensorboard 
* OpenCV

We implement this work CUDA 12.1.

#### How to run it
Similar to UDIS, we also implement this solution in two stages:
* Stage 1 (unsupervised warp): please run "inference.py" in "Warp/Codes/"
* Stage 2 (unsupervised composition): please run "inference.py" in "Composition/Codes/"

## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao. Learning edge-preserved image stitching from multi-scale deep homography[J]. Neurocomputing, 2022, 491: 533-543.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Unsupervised deep image stitching: Reconstructing stitched features to images[J]. IEEE Transactions on Image Processing, 2021, 30: 6184-6197.   
[4] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Deep rectangling for image stitching: a learning baseline[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 5740-5748.   
