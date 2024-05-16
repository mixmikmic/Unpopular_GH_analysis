from __future__ import division, print_function
from google.protobuf import json_format
from grpc.beta import implementations
from sklearn.preprocessing import OneHotEncoder
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os
import sys
import threading
import time
import numpy as np
import tensorflow as tf

SERVER_HOST = "localhost"
SERVER_PORT = 9000

DATA_DIR = "../../data"
TEST_FILE = os.path.join(DATA_DIR, "mnist_test.csv")

IMG_SIZE = 28

MODEL_NAME = "keras-mnist-fcn"

def parse_file(filename):
    xdata, ydata = [], []
    fin = open(filename, "rb")
    i = 0
    for line in fin:
        if i % 10000 == 0:
            print("{:s}: {:d} lines read".format(os.path.basename(filename), i))
        cols = line.strip().split(",")
        ydata.append(int(cols[0]))
        xdata.append(np.reshape(np.array([float(x) / 255. for x in cols[1:]]), 
                     (IMG_SIZE * IMG_SIZE, )))
        i += 1
    fin.close()
    print("{:s}: {:d} lines read".format(os.path.basename(filename), i))
    X = np.array(xdata, dtype="float32")
    y = np.array(ydata, dtype="int32")
    return X, y

Xtest, ytest = parse_file(TEST_FILE)
print(Xtest.shape, ytest.shape)

channel = implementations.insecure_channel(SERVER_HOST, SERVER_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
labels, predictions = [], []
for i in range(Xtest.shape[0]):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = "predict"

    Xbatch, ybatch = Xtest[i], ytest[i]
    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(Xbatch, shape=[1, Xbatch.size]))

    result = stub.Predict(request, 10.0)
    result_json = json.loads(json_format.MessageToJson(result))
    y_ = np.array(result_json["outputs"]["scores"]["floatVal"], dtype="float32")
    labels.append(ybatch)
    predictions.append(np.argmax(y_))

print("Test accuracy: {:.3f}".format(accuracy_score(labels, predictions)))
print("Confusion Matrix")
print(confusion_matrix(labels, predictions))



