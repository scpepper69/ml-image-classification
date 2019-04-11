# Japanese HIRAGANA Prediction Application


## Application Image

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1ff-ZQe95G--1t4MAL7l4EVcQR0nGanIV">


## How to deploy

### Preparation

Please check parent folder.

### Usage
1. docker run

   ```
   # ml/tensorflow/hiragana/app/docker.sh
   ```

1. Startup app.py

   ```
   # nohup python ml/tensorflow/hiragana/app/app.py &
   ```

   
## Architecture

- Learning Model : Tensorflow & Keras

- Model Server : GraphPipe

- Application : Python (Flask)