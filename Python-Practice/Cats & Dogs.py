import torch
from torch import nn

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

import os
def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(C:\Users\Zahid Ullah\Desktop\CatandDog\Images):
    print(f"There are {len(CatandDog)} directories and {len(Images)} images in '{C:\Users\Zahid Ullah\Desktop\CatandDog\Images}'.")
    
image_path = "C:\Users\Zahid Ullah\Desktop\CatandDog\Images"
walk_through_dir(C:\Users\Zahid Ullah\Desktop\CatandDog\Images)
                 
train_dir = "C:\Users\Zahid Ullah\Desktop\CatandDog\Images\train"
test_dir = "C:\Users\Zahid Ullah\Desktop\CatandDog\Images\test"
train_dir, test_dir

import random
from PIL import Image
import glob
from pathlib import Path

# Set seed
random.seed(42) 

# 1. Get all image paths (* means "any combination")
image_path_list= glob.glob(f"{image_path}/*/*/*/*.jpg")

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = Path(random_image_path).parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img