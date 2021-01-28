import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import os
import tensorflow as tf

trained_checkpoint_prefix = 'model/model.ckpt-49491'
export_dir = os.path.join('.', '1')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         strip_default_attrs=True)
    builder.save() 

saved_model_dir = os.path.join(".","1")
converter  = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()
open("EAST.tflite", 'wb').write(tflite_model)
