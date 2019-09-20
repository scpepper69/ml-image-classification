#------------------------------------------------------------------------------
# for Image Classification
# Copyright scpepper
#------------------------------------------------------------------------------

from datetime import datetime
from model import cnn_model, cnn_vgg16, cnn_w_dropout, cnn_w_batchnorm, resnet_v1, resnet_v2 
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Prepare environment
#----------------------------------------------------------------
# Dataset Name
dataset_name='gface64x64'

# Number of Classes
num_classes = 6

# Number of files in each class
num_images = 80

# Image Size
height, width, color = 64, 64, 3

# Specufy Model Structure (CNN, VGG16, RESNET1 or RESNET2)
model_opt="RESNET2"

exec_date = datetime.now().strftime("%Y%m%d%H%M%S")

import os, shutil
#from google.colab import drive
#drive.mount('/content/drive/')
gdrive_base='D:/20.programs/github/ml-image-classification/general/learning/'

# Directory for TensorBorad Logs
log_dir=gdrive_base+'logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Directory for Checkpoint and Froze Model
model_dir=gdrive_base+'models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#----------------------------------------------------------------
# Prepare Dataset
#----------------------------------------------------------------
import numpy as np
import os
import cv2
from glob import glob

# Prepare empty array
ary = np.zeros([num_classes, num_images, height, width, color], dtype=np.int)

# Specify Dataset directory
dir_name='datasets/'+dataset_name

# Specify Class Difinition
labels = np.array([
        'rx-178',
        'msz-006',
        'rx-93',
        'ms-06',
        'rx-78-2',
        'f91'])

c0=0 # rx-178:mk2
c1=0 # msz-006:Z
c2=0 # rx-93:Nu
c3=0 # ms-06:Zaku
c4=0 # rx-78-2:first
c5=0 # f91

# Convert Image Data to Tensor
for file in glob(gdrive_base + dir_name + '/*.jpg'):
    img = cv2.imread(file,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if 'rx-178' in file:
        ary[0, c0] = img
        c0 += 1
    elif 'msz-006' in file:
        ary[1, c1] = img
        c1 += 1
    elif 'rx-93' in file:
        ary[2, c2] = img
        c2 += 1
    elif 'ms-06' in file:
        ary[3, c3] = img
        c3 += 1
    elif 'rx-78-2' in file:
        ary[4, c4] = img
        c4 += 1
    elif 'f91' in file:
        ary[5, c5] = img
        c5 += 1

# Save as npz
np.savez_compressed(dataset_name+'.npz', ary)

# Restore from npz
#ary = np.load(dataset_name+'.npz')['arr_0']

# 画像データのテンソルをソートし、ラベル用テンソルを用意
X_train = np.zeros([num_classes * num_images, height, width, color], dtype=np.int)
for i in range(num_classes):
    for j in range(num_images):
        X_train[(i * num_images) + j] = ary[i][j]

# X_trainはクラス番号でソートされて格納されているので、下記だけでラベルデータが生成できる
Y_train = np.repeat(np.arange(num_classes), num_images)

from sklearn.model_selection import train_test_split

# 検証データの割合を指定
validate_rate=0.2

# 学習データと検証データに分割
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=validate_rate)

print(x_train.shape)
print(x_test.shape)



from tensorflow.keras.utils import to_categorical
import numpy as np

# ラベルデータをone-hot表現へ変換
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 画像データの型を変換
x_train = x_train.reshape(-1, height, width, color).astype(np.float32)
x_test = x_test.reshape(-1, height, width, color).astype(np.float32)
input_shape = (height, width, color)

#----------------------------------------------------------------
# Build Model
#----------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# model構築
if model_opt=="VGG16":
    model = cnn_vgg16(input_shape=input_shape, num_classes=num_classes)
elif model_opt=="RESNET1":
    model = resnet_v1(input_shape=input_shape, num_classes=num_classes)
elif model_opt=="RESNET2":
    model = resnet_v2(input_shape=input_shape, num_classes=num_classes)
else:
#    model=cnn_model(input_shape=input_shape, num_classes=num_classes)
#    model=cnn_w_dropout(input_shape=input_shape, num_classes=num_classes)
    model=cnn_w_batchnorm(input_shape=input_shape, num_classes=num_classes)

from tensorflow.keras import optimizers
from tensorflow.keras import losses
# モデルの学習方法について指定しておく
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Nadam(lr=0.001), metrics=['accuracy'])


# TensorBoardでの可視化のため、出力先の設定
from tensorflow.keras import callbacks

tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_images=1)

# チェックポイント出力先
#RUN = RUN + 1 if 'RUN' in locals() else 1
#checkpoint_path = model_dir + f'run{RUN}/' + model_opt + "_cp-{epoch:04d}.ckpt"
checkpoint_path = f"{model_dir}{dataset_name}_{model_opt}_{exec_date}" + "_cp-{epoch:04d}.ckpt"


checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_cb = callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)

# モデルのサマリ情報の表示
model.summary()

# チェックポイントから学習済みパラメータを復元
#model.load_weights(f'{model_dir}run1/{model_structure}_{data_set}_cp-0010.ckpt')

#----------------------------------------------------------------
# Training the Model
#----------------------------------------------------------------
from keras.preprocessing import image

# epoch数を指定
epochs=10

# batchサイズを指定
batch_size=500

# 学習の実行(fit)
#result = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[tb_cb, cp_cb], validation_data=(x_test, y_test))

# 画像データ生成器を作成する。
params = {
    'rotation_range': 20,
    'zoom_range': 0.10,
    'height_shift_range': 0.1,
    'width_shift_range': 0.1
}
datagen = image.ImageDataGenerator(**params)
datagen.fit(x_train)

# 学習の実行 (fit_generator)
result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=x_train.shape[0], epochs=epochs, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


result.history.keys() # ヒストリデータのラベルを見てみる
print(epochs)
print(result.history['acc'])
plt.plot(range(1, epochs+1), result.history['acc'], label="training")
plt.plot(range(1, epochs+1), result.history['val_acc'], label="validation")
plt.title('Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.xlim([1,epochs])
plt.ylim([0,1])
plt.show()
#plt.savefig(model_opt+'_'+'acc.png')
plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_acc.png")


dataset_name='gface64x64'

plt.plot(range(1, epochs+1), result.history['loss'], label="training")
plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation")
plt.title('Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xlim([1,epochs])
plt.ylim([0,20])
plt.show()
#plt.savefig(model_opt+'_'+'loss.png')
plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_loss.png")

classes = model.predict(x_test, batch_size=128, verbose=1)
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#print(classification_report(np.argmax(y_test, 1), np.argmax(classes, 1)))
#print(confusion_matrix(np.argmax(y_test, 1), np.argmax(classes, 1)))

cmatrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(classes, 1))
cmatrix_plt = pd.DataFrame(cmatrix, index=labels, columns=labels)

plt.figure(figsize = (10,7))
sns.heatmap(cmatrix_plt, annot=True, cmap="Reds", fmt="d")
plt.show()
plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_confusion_matrix.png")

# Keras形式でモデルを出力
output_keras_name = f"{model_dir}{dataset_name}_{model_opt}_{epochs}_{exec_date}_frozen_graph.h5"
model.save(output_keras_name, include_optimizer=False)

# TensorFlow Saved Model形式でモデルを出力
from tensorflow.contrib import saved_model

out_tf_saved_model = f"{model_dir}{dataset_name}_{model_opt}_{epochs}_{exec_date}_saved_models"

if os.path.exists(out_tf_saved_model):
    shutil.rmtree(out_tf_saved_model)
saved_model_path = saved_model.save_keras_model(model, out_tf_saved_model)
