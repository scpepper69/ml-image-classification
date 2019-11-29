#------------------------------------------------------------------------------
# Image Classification Model Builder
# Copyright (c) 2019, scpepper All rights reserved.
#------------------------------------------------------------------------------
import os, shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from glob import glob

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.contrib import saved_model
from keras.preprocessing import image as keras_image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import config as local_conf
from model import cnn_model, cnn_vgg16, cnn_w_dropout, cnn_w_batchnorm, resnet_v1, resnet_v2 

#----------------------------------------------------------------
# Prepare environment
#----------------------------------------------------------------
# from config.py settings
#gdrive_base=local_conf.gdrive_base
#dataset_name=local_conf.dataset_name
#num_classes = local_conf.num_classes
#labels = local_conf.labels
#num_images = local_conf.num_images
#height= local_conf.height
#width= local_conf.width
#color= local_conf.color
#model_opt=local_conf.model_opt
#validate_rate=local_conf.validate_rate
#epochs=local_conf.epochs
#batch_size=local_conf.batch_size

def main(gdrive_base, dataset_name, num_classes, labels, num_images, width, height, color, model_opt, validate_rate=0.2, epochs=20, batch_size=4):

    exec_date = datetime.now().strftime("%Y%m%d%H%M%S")

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
    # Prepare empty array
    ary = np.zeros([num_classes, num_images, height, width, color], dtype=np.int)
    counters = np.zeros(num_classes, dtype=np.int)

    # Specify Dataset directory
#    dir_name='datasets/'+dataset_name
    dir_name='datasets/'

    # Convert Image Data to Tensor
    for file in glob(gdrive_base + dir_name + '/*.jpg'):
        if color==1:
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        else:
            print(color)
            img = cv2.imread(file,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(len(labels)):
                if "/"+labels[i] in file:
                    ary[i, counters[i]] = img
                    counters[i] += 1

    # Save as npz
    np.savez_compressed(f"{gdrive_base}{dir_name}np.npz", ary)

    # Restore from npz
    #ary = np.load(f"{gdrive_base}{dir_name}.npz")['arr_0']

    # Sort train tensor for generating answer tensor 
    X_train = np.zeros([num_classes * num_images, height, width, color], dtype=np.int)
    for i in range(num_classes):
        for j in range(num_images):
            X_train[(i * num_images) + j] = ary[i][j]

    # Generate answer tensor
    Y_train = np.repeat(np.arange(num_classes), num_images)

    # Split the data 
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=validate_rate)

    # Convert answer tensor to "one-hot"
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Convert the image shape
    x_train = x_train.reshape(-1, height, width, color).astype(np.float32)
    x_test = x_test.reshape(-1, height, width, color).astype(np.float32)
    input_shape = (height, width, color)

    #----------------------------------------------------------------
    # Build Model
    #----------------------------------------------------------------
    # for resolve "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED" error.
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    K.set_session(sess)

    # Building model
    if model_opt=="VGG16":
        model = cnn_vgg16(input_shape=input_shape, num_classes=num_classes)
    #elif model_opt=="RESNET1":
    #    model = resnet_v1(input_shape=input_shape, num_classes=num_classes)
    elif model_opt=="RESNET":
        model = resnet_v2(input_shape=input_shape, num_classes=num_classes)
    else:
    #    model=cnn_model(input_shape=input_shape, num_classes=num_classes)
    #    model=cnn_w_dropout(input_shape=input_shape, num_classes=num_classes)
        model=cnn_w_batchnorm(input_shape=input_shape, num_classes=num_classes)

    # Compile Model
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Nadam(lr=0.001), metrics=['accuracy'])

    # Callback setting for TensorBoard
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_images=1)

    # Checkpoint setting
    checkpoint_path = f"{model_dir}{dataset_name}_{model_opt}_{exec_date}" + "_cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Callback for checkpoint
    cp_cb = callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)

    # Show model summary
    model.summary()

    # Restore learned parameters from checkpoint
    #model.load_weights(f'{model_dir}run1/{model_structure}_{data_set}_cp-0010.ckpt')

    #----------------------------------------------------------------
    # Training the Model
    #----------------------------------------------------------------
    # Data generator parameter setting
    params = {
        'rotation_range': 20,
        'zoom_range': 0.10,
        'height_shift_range': 0.1,
        'width_shift_range': 0.1
    }
    datagen = keras_image.ImageDataGenerator(**params)
    datagen.fit(x_train)

    from random import shuffle
    from scipy import ndimage

    def generator(x, y1, train):

        while True:
            if train:
                keys = list(range(len(x)))
                shuffle(keys)
            else:
                keys = list(range(len(y1)))
                shuffle(keys)
            inputs = []
            label1 = []

            for key in keys:
                img = x[key]
                if train:
                    # 画像の回転
                    rotate_rate = np.random.normal(0,0.5)*10
                    img = ndimage.rotate(x[key], rotate_rate)
                    img = cv2.resize(img,(width, height))
                    # 画像のぼかし
                    if np.random.randint(0,2):
                        filter_rate = np.random.randint(0,6)
                        img = ndimage.gaussian_filter(img, sigma=filter_rate)

                inputs.append(img)
                label1.append(y1[key])

                if len(inputs) == batch_size:
                    tmp_inputs = np.array(inputs)
                    tmp_label1 = np.array(label1)
                    inputs = []
                    label1 = []
                    yield tmp_inputs, {'dense': tmp_label1}

    # 学習の実行 (fit_generator)
    """
    result = model.fit_generator(generator(x_train, y_train, True),
                                steps_per_epoch=x_train.shape[0], 
                                epochs=epochs, 
                                validation_data=generator(x_test, y_test, False), 
                                validation_steps=2, 
                                verbose=1)
    #                             callbacks=[tb_cb])
    """

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Execute training
    #result = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[tb_cb, cp_cb], validation_data=(x_test, y_test))
    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0], epochs=epochs, validation_data=(x_test, y_test), callbacks=[early_stopping])

    # Evaluate the training score
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Show accuracy graph
    result.history.keys() 
    print(epochs)
    #plt.plot(range(1, epochs+1), result.history['acc'], label="training")
    #plt.plot(range(1, epochs+1), result.history['val_acc'], label="validation")
    plt.plot(result.history['acc'], label="training")
    plt.plot(result.history['val_acc'], label="validation")
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.xlim([1,epochs])
    plt.ylim([0,1])
    #plt.show()
    plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_acc.png")
    plt.figure()

    # Show loss graph
    #plt.plot(range(1, epochs+1), result.history['loss'], label="training")
    #plt.plot(range(1, epochs+1), result.history['val_loss'], label="validation")
    plt.plot(result.history['loss'], label="training")
    plt.plot(result.history['val_loss'], label="validation")
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.xlim([1,epochs])
    plt.ylim([0,20])
    #plt.show()
    plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_loss.png")
    plt.figure()

    # Predict validation data
    classes = model.predict(x_test, batch_size=128, verbose=1)

    # Show confusion matrix
    cmatrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(classes, 1))
    cmatrix_plt = pd.DataFrame(cmatrix, index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(cmatrix_plt, annot=True, cmap="Reds", fmt="d")
    #plt.show()
    plt.savefig(f"{model_dir}{dataset_name}_{model_opt}_{exec_date}_confusion_matrix.png")

    # Output model as keras format
    output_keras_name = f"{model_dir}{dataset_name}_{model_opt}_{epochs}_{exec_date}_frozen_graph.h5"
    model.save(output_keras_name, include_optimizer=False)
    print("Saved Keras Model.")

    output_tflite_name = f"{model_dir}{dataset_name}_{model_opt}_{epochs}_{exec_date}_frozen_graph.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model_file(output_keras_name)
    #converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=model.inputs, output_tensors=model.outputs) 
    try:
        tflite_model = converter.convert()
    except:
        import traceback
        traceback.print_exc()
    open(output_tflite_name, "wb").write(tflite_model)
    print("Saved TFLite Model.")

    # Output model as tensorflow saved model format
    #out_tf_saved_model = f"{model_dir}{dataset_name}_{model_opt}_{epochs}_{exec_date}_saved_models"
    #if os.path.exists(out_tf_saved_model):
    #    shutil.rmtree(out_tf_saved_model)
    #saved_model_path = saved_model.save_keras_model(model, out_tf_saved_model)

    return score[1], score[0]
