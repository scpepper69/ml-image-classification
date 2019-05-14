# Japanese KANJI & HIRAGANA Prediction Application

Sample application is published on [my blog](https://www.scpepper.tokyo/2019/04/18/post-313/).



## Application Image

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1wUoAll87EDuZBlIgvRFM2uC9Mpf5g8oc">


## How to deploy

### Preparation

Please see README.md at parent directory.

### Usage
1. docker run

   ```bash
   cd ml-image-classification/japanese/app/
   sh ./docker.sh
   ```

1. Startup japanese.py

   ```bash
   cd ./app
   nohup python ./japanese.py &
   ```
   
   
## Application Architecture

- Learned Model : TensorFlow & Keras
- Model Server : GraphPipe
- Web Application : Python (Flask)



## Model Structure

I used the all characters in dataset [ETL8G](http://etlcdb.db.aist.go.jp/?page_id=2461&lang=ja). 

Model structure is built by CNN. Please see learn_japanese_batchnorm.py for details.

