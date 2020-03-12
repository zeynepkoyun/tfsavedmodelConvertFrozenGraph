
import tensorflow as tf
import numpy as np

from PIL import Image
from keras.preprocessing.image import img_to_array


image_name = './00000000.jpg'

PATH_TO_CKPT = "./frozen_graph.pb"


def main():
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('data:0')
    score = detection_graph.get_tensor_by_name('flatten_1/Reshape:0')

    image = Image.open(image_name)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image1 = img_to_array(image)
    img = np.expand_dims(image1, axis=0)
    img = preprocess_input(img)

    embed = sess.run(
        label,
        feed_dict={image_tensor: img})

    return embed[0] 

if __name__ == '__main__':
    main()
