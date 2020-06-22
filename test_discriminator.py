import os

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import tensorflow as tf
import glob
import dataset
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
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    #tensor = np.zeros((1, 3, 512, 512))
    
    for filename in os.listdir('/home/jxh3105/SSL-PG-GAN/CatVDog/PetImages/Dog10P/'):
        print(filename)
        img = np.asarray(PIL.Image.open('/home/jxh3105/SSL-PG-GAN/CatVDog/PetImages/Dog10P/DOG-99_2_3.jpg')).reshape(3,512,512)
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        scores, labels = test_discriminator(8, img)
        results = labels.eval()
        print(results)
        if results[0][0] >= results[0][1]:
            print("DOG")
        else:
            print("CAT")
        print("Labels",labels.eval())
        print("Scores",scores.eval())


