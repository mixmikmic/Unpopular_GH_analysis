from __future__ import division, print_function
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy
import os
import shutil
import tensorflow as tf

DATA_DIR = "../../data"
TEST_FILE = os.path.join(DATA_DIR, "mnist_test.csv")

IMG_SIZE = 28
BATCH_SIZE = 128
NUM_CLASSES = 10

MODEL_DIR = os.path.join(DATA_DIR, "01-tf-serving")
TF_MODEL_NAME = "model-5"

EXPORT_DIR = os.path.join(DATA_DIR, "tf-export")
MODEL_NAME = "ktf-mnist-cnn"
MODEL_VERSION = 1

tf.contrib.keras.backend.set_learning_phase(0)
sess = tf.contrib.keras.backend.get_session()
with sess.as_default():
    saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, TF_MODEL_NAME + ".meta"))
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

shutil.rmtree(EXPORT_DIR, True)

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {"x": tf.FixedLenFeature(shape=[IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
X = tf.identity(tf_example["x"], name="X")
Y = tf.placeholder("int32", shape=[None, 10], name="Y")
Y_ = tf.placeholder("int32", shape=[None, 10], name="Y_")

export_dir = os.path.join(EXPORT_DIR, MODEL_NAME)
export_path = os.path.join(export_dir, str(MODEL_VERSION))
print("Exporting model to {:s}".format(export_path))

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
tensor_info_y = tf.saved_model.utils.build_tensor_info(Y)
tensor_info_y_ = tf.saved_model.utils.build_tensor_info(Y_)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"images": tensor_info_x,
                "labels": tensor_info_y},
        outputs={"scores": tensor_info_y_},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
print(prediction_signature)

legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map = {
        "predict": prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature
    },
    legacy_init_op=legacy_init_op)
builder.save()



