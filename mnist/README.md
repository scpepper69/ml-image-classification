# MNIST Prediction Application



Sample application is published on [my blog](https://www.scpepper.tokyo/2019/01/16/post-244/).



## Application Image

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1-k8yK38LhMMVJ4ZZt4osTjPSafMR6xNU">


## How to deploy

### Preparation

Please see README.md at parent directory.

### Usage
1. docker run

   ```bash
   # cd ml-image-classification/mnist/app/
   # ./app/docker.sh
   ```

1. Startup mnist.py

   ```bash
   # cd ./app
   # nohup python ./mnist.py &
```
   
   
## Architecture

- Learning Model : TensorFlow & Keras
- Model Server : GraphPipe
- Web Application : Python (Flask)



## Model Structure

The models are based on TensorFlow Tutorials.

- beginner-graph.pb
- expert-graph.pb

Model structure is built by CNN. Please see mnist_test_beginner.py and mnist_test_expert.py for details.

