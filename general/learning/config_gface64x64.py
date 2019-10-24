#------------------------------------------------------------------------------
# Image Classification Model Builder
# Copyright (c) 2019, scpepper All rights reserved.
#------------------------------------------------------------------------------
import numpy as np

# Please change these values for your environment.
# -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> ->

# Specify root Directory
gdrive_base='D:/20.programs/github/ml-image-classification/general/learning/'

# Specufy dataset name
dataset_name='gface64x64'

# Specufy number of classes
num_classes = 6

# Specify class names
labels = np.array([
        'rx-178',
        'msz-006',
        'rx-93',
        'ms-06',
        'rx-78-2',
        'f91'])

# Specufy number of files in each class
num_images = 80

# Specufy image size
height, width, color = 64, 64, 3

# Specufy Model Structure (CNN, VGG16, RESNET1 or RESNET2)
model_opt="RESNET2"

# Specify the rate of validation
validate_rate=0.2

# Specify training epoches
epochs=10

# Specify training batch size
batch_size=32

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <-
