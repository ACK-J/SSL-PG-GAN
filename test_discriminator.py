import os

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import tensorflow as tf
import glob
import dataset
import sys
import config
import tfutil
import misc
from train import process_reals
import PIL.Image

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def test_discriminator(run_id, tensor, snapshot=None):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)
    scores, labels = fp32(D.get_output_for(tensor, is_training=False))
    return (scores, labels)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print("Error Wrong Arguments: python3 test_discriminator.py <path to out of sample images> <id of training "
              "round>\npython3 test_discriminator.py /home/user/OutOfSample/Images/ 2")
        exit(1)
    if not os.path.isdir(args[1]):
        print("Error: " + args[1] + " does not exist")
        exit(1)
    # Checking to see if there is a / at the end of the string entered
    if args[1][-1] != '/':
        args[1] = args[1] + "/"
    
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    #tensor = np.zeros((1, 3, 512, 512))

    for filename in os.listdir(args[1]):
        print(filename)
        img = np.asarray(PIL.Image.open(args[1] + filename)).reshape(3,512,512)
        img = np.expand_dims(img, axis=0)
        print(img)
        scores, labels = test_discriminator(args[2], img)

        print("Labels",labels.eval())
        print("Scores",scores.eval())



