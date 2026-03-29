### `scripts/check_environment.py`
import sys

print("Python version:", sys.version)

import torch
import torchvision
import cv2
import numpy
import matplotlib

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)

print("CUDA available:", torch.cuda.is_available())
print("Environment check passed.")