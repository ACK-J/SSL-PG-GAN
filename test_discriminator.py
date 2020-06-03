import os
import numpy as np
import tensorflow as tf
import glob
import dataset
import config
import tfutil
import misc
from train import process_reals
import PIL.Image



def error(msg):
    print('Error: ' + msg)
    exit(1)

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert (os.path.isdir(self.tfrecord_dir))

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2 ** self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            # quant = np.rint(img).clip(0, 255).astype(np.uint8)
            quant = img.astype(np.uint8)
            # Converting the np array to a tensor
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        print("cur", self.cur_images)
        print("shape", labels.shape)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_from_images(testing_tfrecord_dir, testing_dir, shuffle):
    print('Loading images from "%s"' % testing_dir)
    testing_filenames = sorted(glob.glob(os.path.join(testing_dir, '*')))
    if len(testing_filenames) == 0:
        error('No input images found in ' + testing_dir)

    img = np.asarray(PIL.Image.open(testing_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    labels = [3] * len(testing_filenames)
    labels = np.array(labels)
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    #  Adding labeled data
    with TFRecordExporter(testing_tfrecord_dir, len(testing_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(testing_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(testing_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose(2, 0, 1)  # HWC => CHW
            tfr.add_image(img)
        tfr.add_labels(onehot[order])

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def use_discriminator(run_id, testing_dataset, snapshot=None):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    imported_graph = tf.get_default_graph()
    graph_op = imported_graph.get_operations()
    with open('output.txt', 'w') as f:
        for i in graph_op:
            f.write(str(i))

    scores, labels = fp32(D.get_output_for(testing_dataset, is_training=False))
    return (scores, labels)


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
    tensor = np.zeros((1, 3, 512, 512))
    scores, labels = test_discriminator(3, tensor)
    print(labels.eval())

    exit()
    # Make sure that all the images in this directory are square and
    # a power of 2. ex) 512X512 or 32X32
    data_dir = "CatVDog/PetImages/Cat10P"
    RUN_ID = 37

    #  Takes in a directory full of square images
    #create_from_images("testing", data_dir, True)

    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    #  Loads it into a dataset
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.testingdataset)
    mirror_augment = False
    drange_net = [-1, 1]

    #  Processes the dataset
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            scores, labels = use_discriminator(RUN_ID, reals_gpu, snapshot=None)
            real_scores_out = tfutil.autosummary('Loss/real_scores', scores)
            lodin_placeholder = tf.get_default_graph().get_tensor_by_name('Inputs/lod_in:0')
            print(real_scores_out.eval(feed_dict={lodin_placeholder: 4}))
            #tf.print(real_scores_out)



