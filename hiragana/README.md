# Japanese HIRAGANA Prediction Application

Sample application is published on [my blog](https://www.scpepper.tokyo/2019/01/16/post-244/).


## Application Image

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1ff-ZQe95G--1t4MAL7l4EVcQR0nGanIV">


## How to deploy

### Preparation

Please see README.md at parent directory.

### Usage
1. docker run

   ```bash
   cd ml-image-classification/hiragana/app/
   sh ./docker.sh
   ```

1. Startup hiragana.py

   ```bash
   cd ./app
   nohup python ./hiragana.py &
   ```
   
   
## Application Architecture

- Learned Model : TensorFlow & Keras
- Model Server : GraphPipe
- Web Application : Python (Flask)



## Model Structure

I used the dataset [ETL8G](http://etlcdb.db.aist.go.jp/?page_id=2461&lang=ja) and extracted only HIRAGANA characters by read_hiragana.py .

Model structure is built by CNN. Please see learn_hiragana.py for details.


