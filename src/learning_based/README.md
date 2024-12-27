
## Code
#### Requirement
* numpy==1.26.3
* pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
* scikit-image==0.25.0
* tensorboard==2.18.0
* opencv-python==4.10.0.84

We implement this work CUDA 12.1.

#### How to run it
Similar to UDIS, we also implement this solution in two stages:
* Stage 1 (unsupervised warp): please run "inference.py" in "Warp/Codes/"
* Stage 2 (unsupervised composition): please run "inference.py" in "Composition/Codes/"
  
