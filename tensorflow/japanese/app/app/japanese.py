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

import urllib.request
#import helper.directory
import json

japanese = []
with open("./japanese_list.txt", "r") as f:
    for line in f:
        japanese.append(line.rstrip('\n'))

def hira(n):
    return japanese[n]

app = Flask(__name__)
CORS(app) # ローカルへAjaxでPOSTするため

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ans,yomi,t1,t2,t3 = get_answer(request)
#        return jsonify({'ans': ans})
        return jsonify({'ans': ans, 'yomi': yomi, 't1': t1, 't2': t2, 't3': t3})
    else:
        return render_template('index.html')

def result(img):
#    K.clear_session() # セッションを毎回クリア
#    model = load_model(os.path.abspath(os.path.dirname(__file__)) + '/model.h5')
#    x = np.expand_dims(x, axis=0)
#    x = x.reshape(x.shape[0],28,28,1)
#    img = img.reshape(1, 1024)
    img = img.reshape(1, 32, 32, 1)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    pred = remote.execute("http://localhost:9012", img)
    r = np.argmax(pred, axis=1)
    pp = pred*100
#    top1 = str(np.argsort(-pp)[0][0])+ " (" +str(int(np.sort(-pp)[0][0]*-1))+"%)"
    top1 = japanese[np.argsort(-pp)[0][0]]+ " (" +str(int(np.sort(-pp)[0][0]*-1))+"%)"
    top2 = japanese[np.argsort(-pp)[0][1]]+ " (" +str(int(np.sort(-pp)[0][1]*-1))+"%)"
    top3 = japanese[np.argsort(-pp)[0][2]]+ " (" +str(int(np.sort(-pp)[0][2]*-1))+"%)"
    print(top1)
#    return int(r)
    ret = japanese[np.argsort(-pp)[0][0]]
    return ret,top1,top2,top3

def get_answer(req):
    img_str = re.search(r'base64,(.*)', req.form['img']).group(1)
    nparr = np.fromstring(base64.b64decode(img_str), np.uint8)
    img_src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_negaposi = 255 - img_src
#    img_negaposi = img_src
    img_gray = cv2.cvtColor(img_negaposi, cv2.COLOR_BGR2GRAY)
#    img_resize = cv2.resize(img_gray,(28,28))
    img_resize = cv2.resize(img_gray,(32,32))
    cv2.imwrite(f"images/{datetime.now().strftime('%s')}.jpg",img_resize)
    print('ZZ')
    ans,t1,t2,t3 = result(img_resize)
    try:
        yomi = str(get_yomi(ans))
        print(yomi)
    except:
        yomi = " "
#    return int(ans),t1,t2,t3
    return ans,yomi,t1,t2,t3

def get_yomi(character):
    # convert character to he x unicode
    letter_a = str(character)
    decimal_a = ord(letter_a)
    hex_A = hex(decimal_a)

    # insert into api request format
    request_url = "https://mojikiban.ipa.go.jp/mji/q?UCS=*"
    request_url = request_url.replace('*', hex_A)

    req = urllib.request.Request(request_url)

    with urllib.request.urlopen(req) as res:
        body = json.load(res)

    return body['results'][0]['読み']

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9013)

