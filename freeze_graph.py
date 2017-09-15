import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

def freeze_graph(model_dir, output_node_names):
    """
    freeze the saved checkpoints/graph to *.pb
    """
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = os.path.join(model_dir, "frozen_graph.pb")
    
    saver = tf.train.import_meta_graph(input_checkpoint + ".meta", 
                                       clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     input_graph_def,
                                                                     output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph" % (len(output_graph_def.node)))