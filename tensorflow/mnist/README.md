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
   # ./app/docker.sh
   ```

1. Startup app.py

   ```bash
   # nohup python ./app/app/app.py &
   ```

   
## Architecture

- Learning Model : Tensorflow & Keras
- Model Server : GraphPipe
- Application : Python (Flask)

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1lT1dl5usZaU0laE9H1ig9tPpetn6sMiI">