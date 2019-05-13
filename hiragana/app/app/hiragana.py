from datetime import datetime
import cv2
import re
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
import os,sys
import requests
from graphpipe import remote
from matplotlib import pylab as plt

hiragana = []
with open("./hiragana_list.txt", "r") as f:
    for line in f:
        hiragana.append(line)

def hira(n):
    return hiragana[n]

app = Flask(__name__)
CORS(app) # To Post by Ajax

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ans,t1,t2,t3 = get_answer(request)
        return jsonify({'ans': ans, 't1': t1, 't2': t2, 't3': t3})
    else:
        return render_template('index.html')

def result(img):
    img = img.reshape(1, 32, 32, 1)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    pred = remote.execute("http://localhost:9002", img)
    r = np.argmax(pred, axis=1)
    pp = pred*100
    top1 = hiragana[np.argsort(-pp)[0][0]]+ " (" +str(int(np.sort(-pp)[0][0]*-1))+"%)"
    top2 = hiragana[np.argsort(-pp)[0][1]]+ " (" +str(int(np.sort(-pp)[0][1]*-1))+"%)"
    top3 = hiragana[np.argsort(-pp)[0][2]]+ " (" +str(int(np.sort(-pp)[0][2]*-1))+"%)"
    print(top1)
    ret = hiragana[np.argsort(-pp)[0][0]]
    return ret,top1,top2,top3

def get_answer(req):
    img_str = re.search(r'base64,(.*)', req.form['img']).group(1)
    nparr = np.fromstring(base64.b64decode(img_str), np.uint8)
    img_src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_negaposi = 255 - img_src
    img_gray = cv2.cvtColor(img_negaposi, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray,(32,32))
    cv2.imwrite(f"images/{datetime.now().strftime('%s')}.jpg",img_resize)
    print('ZZ')
    ans,t1,t2,t3 = result(img_resize)
    return ans,t1,t2,t3

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8002)
