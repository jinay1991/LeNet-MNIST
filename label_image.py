import tensorflow as tf
import cv2
import numpy as np

def load_graph(frozen_graph_filename):
    """
    Loads Frozen graph
    """
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def load_labels(label_file):
    """
    Load Labels (labels.txt)
    """
    label = []
    with open(label_file, "r") as f:
        for line in f:
            label.append(line)
    return label

def label_image(image, model, labels):
    """
    Label Image
    """
    graph = load_graph(model)

    for op in graph.get_operations():
        print(op.name)

    input_operation = graph.get_operation_by_name('import/input')
    output_operation = graph.get_operation_by_name('import/softmax_tensor')

    file_reader = tf.read_file(image, "file_reader")
    image_reader = tf.image.decode_png(file_reader, channels = 3)
    image_reader = tf.image.rgb_to_grayscale(image_reader)
    int_caster = tf.cast(image_reader, tf.int32)
    dims_expander = tf.expand_dims(int_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [28, 28])
    with tf.Session() as sess:
        result = sess.run(resized)

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(output_operation.outputs[0], 
                               feed_dict={input_operation.outputs[0]:result})
    
    predictions = np.squeeze(predictions[0])
    top_k = predictions.argsort()[-3:][::-1]
    labelsProb = load_labels(labels)
    print("=" * 10)
    print("Results:")
    for i in top_k:
        print("probability: %s %% for label: %s" % (int(predictions[i]) * 100, labelsProb[i]))
    print("=" * 10)

# Self Test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image (input.jpg) path")
    parser.add_argument("--model", help="model (frozen_graph.pb) path")
    parser.add_argument("--labels", help="labels (labels.txt) path")
    args = parser.parse_args()

    label_image(args.image, args.model, args.labels)