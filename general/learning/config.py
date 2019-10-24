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
dataset_name='carp150x200'

# Specufy number of classes
num_classes = 6

# Specify class names
labels = np.array([
        '1_suzuki',
        '14_ohsera',
        '19_nomura',
        '33_kikuchi',
        '55_matsuyama',
        '95_batista'])

# Specufy number of files in each class
num_images = 60

# Specufy image size
height, width, color = 200, 150, 3

# Specufy Model Structure (CNN, VGG16, RESNET1 or RESNET2)
#model_opt="RESNET2"
model_opt="VGG16"

# Specify the rate of validation
validate_rate=0.2

# Specify training epoches
epochs=10

# Specify training batch size
batch_size=16

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <-
