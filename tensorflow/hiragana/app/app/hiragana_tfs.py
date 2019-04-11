import os  
import time  
import grpc  
import numpy as np  
import tensorflow as tf  
from PIL import Image  
from tensorflow.contrib.util import make_tensor_proto  
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc  

from tensorflow.python.keras.layers import Conv2D, Convolution2D, MaxPooling2D

def predict_number(path, is_parallel):  
    img = np.asarray(Image.open(path), dtype="float32").reshape(32,32,1)  
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)

#    host = os.environ.get("TF_SERVING_HOST")  
    host = "127.0.0.1:8500"  
    channel = grpc.insecure_channel(host)  
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)  
    request = predict_pb2.PredictRequest()  
    request.model_spec.name = "hiragana"  
    request.model_spec.signature_name = "serving_default"  
    request.inputs["input"].CopyFrom(tf.contrib.util.make_tensor_proto(img))  
    start_time = time.time()  
    if is_parallel:  
        futures = []  
        for _ in range(1000):  
            futures.append(stub.Predict.future(request))  
        for future in futures:  
            future.result()  
    else:  
        for _ in range(1000):  
            stub.Predict(request)  
    end_time = time.time()  
    print(f"{end_time - start_time:.2}s")  

def _parse_args():  
    import argparse  
    psr = argparse.ArgumentParser()  
    psr.add_argument("path")  
    psr.add_argument("--parallel", default=False, action="store_true")  
    return psr.parse_args()  

if __name__ == "__main__":  
    args = _parse_args()  

    tf.global_variables_initializer()
    tf.tables_initializer()

    predict_number(args.path, args.parallel)  

