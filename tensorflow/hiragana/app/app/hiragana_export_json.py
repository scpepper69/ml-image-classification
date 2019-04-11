import argparse
import json

from PIL import Image
import numpy as np
import requests
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.preprocessing import image

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
#img = np.asarray(Image.open(image_path), dtype="float32").reshape((1, -1))
img = np.asarray(Image.open(image_path), dtype="float32").reshape(32,32,1)
#img = img.reshape(1, 32, 32, 1)
img = img.astype(np.float32)
img = np.multiply(img, 1.0 / 255.0)

#img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
#img = img.astype('float16')

payload = {
    "signature_name": 'serving_default',
    "inputs": [img.tolist()]
}

with open('./hiragana_test01.json', mode='w') as f:
    f.write(json.dumps(payload))

# sending post request to TensorFlow Serving server
#r = requests.post('http://127.0.0.1:8500/v1/models/hiragana:predict', json=payload)
#pred = json.loads(r.content.decode('utf-8'))

# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
#print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))
