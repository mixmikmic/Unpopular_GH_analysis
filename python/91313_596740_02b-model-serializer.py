from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

import keras.backend as K
from keras.models import load_model
import os
import shutil

K.set_learning_phase(0)

DATA_DIR = "../../data"
EXPORT_DIR = os.path.join(DATA_DIR, "tf-export")

MODEL_NAME = "keras-mnist-fcn"
MODEL_VERSION = 1

MODEL_BIN = os.path.join(DATA_DIR, "{:s}-best.h5".format(MODEL_NAME))
EXPORT_PATH = os.path.join(EXPORT_DIR, MODEL_NAME)

model = load_model(MODEL_BIN)

shutil.rmtree(EXPORT_PATH, True)

full_export_path = os.path.join(EXPORT_PATH, str(MODEL_VERSION))
builder = saved_model_builder.SavedModelBuilder(full_export_path)
signature = predict_signature_def(inputs={"images": model.input},
                                  outputs={"scores": model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})
    builder.save()



