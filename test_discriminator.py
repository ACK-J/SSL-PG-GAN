import os

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import tensorflow as tf
import glob
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

def test_discriminator(D, tensor):
    scores, labels = fp32(D.get_output_for(tensor, is_training=False))
    return (scores, labels)


if __name__ == '__main__':
    args = sys.argv
    givenCorrectClassIndex= False
    #  Checking args
    if len(args) < 4:
        print("Error Wrong Arguments: python3 test_discriminator.py <path to out of sample images> <id of training "
              "round> <pixels> <OPTIONAL: index of correct class> \npython3 test_discriminator.py /home/user/OutOfSample/Images/ 2 512")
        exit(1)
    if len(args) == 5:
    	givenCorrectClassIndex = True


    # Making sure the directory exists
    if not os.path.isdir(args[1]):
        print("Error: " + args[1] + " does not exist")
        exit(1)

    # Checking to see if there is a / at the end of the string entered
    if args[1][-1] != '/':
        args[1] = args[1] + "/"
    
    # Initializing tensorflow
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    #  Deserializing the pickle file to get the Discriminator
    snapshot=None
    run_id = args[2]
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading discriminator from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    # example ndim = 512 for an image that is 512x512 pixels 
    # All images for SSL-PGGAN must be square
    ndim = int(args[3])
    correct=0
    guesses=0

    # Go through every image that needs to be tested
    for filename in os.listdir(args[1]):
    	guesses+=1
    	#tensor = np.zeros((1, 3, 512, 512))
        print(filename)
        img = np.asarray(PIL.Image.open(args[1] + filename)).reshape(3,ndim,ndim)
        img = np.expand_dims(img, axis=0) # makes the image (1,3,512,512)
        scores, labels_out = test_discriminator(D, img)

        correctLabel = int(args[4])

        if givenCorrectClassIndex:
	        sample_probs = tf.nn.softmax(labels_out)
	        label = np.argmax(sample_probs.eval()[0], axis=0)
	        if label == correctLabel:
	            correct += 1
	        print("LABEL: ",label)
	        print("Correct: ", correct, "\n", "Guesses: ", guesses, "\n", "Percent correct: ", (correct/guesses))
