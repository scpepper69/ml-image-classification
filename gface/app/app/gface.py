from datetime import datetime
import cv2
import re
import base64
import numpy as np

from io import BytesIO
from PIL import Image, ImageOps
import os,sys
import requests
from graphpipe import remote
from matplotlib import pylab as plt

import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'bmp'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

gface = []
with open("./gface_list.txt", "r") as f:
    for line in f:
        gface.append(line)

def hira(n):
    return gface[n]

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = './uploads/' + filename
            outfile,ans,t1,t2,t3 = get_answer('./uploads/'+filename)
            return render_template('index.html', img_url=outfile, ans=ans, t1=t1, t2=t2, t3=t3)
#            return render_template('index.html', img_url=img_url, ans=ans, t1=t1, t2=t2, t3=t3)
        else:
            return ''' <p>This extension is not allowed.</p> '''
    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def result(img):
    img = img.reshape(1, 64, 64, 1)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    pred = remote.execute("http://localhost:9004", img)
    r = np.argmax(pred, axis=1)
    pp = pred*100
    top1 = gface[np.argsort(-pp)[0][0]]+ " (" +str(int(np.sort(-pp)[0][0]*-1))+"%)"
    top2 = gface[np.argsort(-pp)[0][1]]+ " (" +str(int(np.sort(-pp)[0][1]*-1))+"%)"
    top3 = gface[np.argsort(-pp)[0][2]]+ " (" +str(int(np.sort(-pp)[0][2]*-1))+"%)"
    print(top1)
    ret = gface[np.argsort(-pp)[0][0]]
    return ret,top1,top2,top3

def get_answer(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)

    orgHeight, orgWidth = img.shape[:2]
    toWidth=256
    toHeight=round(orgWidth/orgHeight*toWidth)

    img_out = cv2.resize(img, dsize=(toHeight,toWidth))

    ofile,oext=os.path.splitext(img_path)
    outfile=ofile+"_"+str(datetime.now().strftime('%s'))+oext
    cv2.imwrite(outfile, img_out)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img, dsize=(64,64))

#    img_src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#    img_negaposi = 255 - img_src
#    img_negaposi = img_src
#    img_gray = cv2.cvtColor(img_negaposi, cv2.COLOR_BGR2GRAY)
#    img_resize = cv2.resize(img_gray,(28,28))
#    img_resize = cv2.resize(img_gray,(64,64))
#    cv2.imwrite(f"images/{datetime.now().strftime('%s')}.jpg",img_resize)
    print('ZZ')
    ans,t1,t2,t3 = result(img_resize)
    return outfile,ans,t1,t2,t3

if __name__ == '__main__':
    app.debug = True
    app.run(debug=False, host='0.0.0.0', port=8004)    

