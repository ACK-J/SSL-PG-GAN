# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import PIL.Image
import tfutil
import dataset


# ----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)


# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()


# ----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))


# ----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__')  # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func):  # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x,
                                   post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)

        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res


# ----------------------------------------------------------------------------

def display(tfrecord_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print('Displaying images')
            import cv2  # pip install opencv-python
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(1, 2, 0)[:, :, ::-1])  # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)


# ----------------------------------------------------------------------------

def extract(tfrecord_dir, output_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        idx += 1
    print('Extracted %d images.' % idx)


# ----------------------------------------------------------------------------

def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset_a = dataset.TFRecordDataset(tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))


def create_from_images(labeled_tfrecord_dir, unlabeled_tfrecord_dir, labeled_dir, unlabeled_dir, shuffle):
    # Checking to see if there is two slashes at the end instead of 1
    if labeled_dir[-1] == "/" and labeled_dir[-2] == "/":
        labeled_dir = labeled_dir[:-1]
    if unlabeled_dir[-1] == "/" and unlabeled_dir[-2] == "/":
        unlabeled_dir = unlabeled_dir[:-1]
     
    # Checking to make sure the path exists    
    if not os.path.isdir(labeled_dir):
        error("Path " + labeled_dir + " does not exist!")
    if not os.path.isdir(unlabeled_dir):
        error("Path " + unlabeled_dir + " does not exist!")
    
    # This lists all of the directories in the provided labeled directory. Each class should have its own folder
    # within this directory. It also prepends the full path before it and makes sure .git isn't included
    classes_dir = [labeled_dir + name for name in os.listdir(labeled_dir) if os.path.isdir(os.path.join(labeled_dir, name)) and name != '.git']
    Num_classes = len(classes_dir)

    labeled_filenames = []
    # Go through each class directory and list all the full paths to each file and store them in an array
    for each_class in classes_dir:
        print('Loading images from "%s"' % each_class)
        labeled_filenames.append(list(sorted(glob.glob(os.path.join(each_class, '*')))))
    # Go through that array and assign Labels to each image
    labels = []
    for i in range(Num_classes):
        labels += [i] * len(labeled_filenames[i])
    print("Number of classes: " + str(len(classes_dir)))

    # Converting labels into np array and one hot encoding it
    labels = np.array(labels)
    onehot = np.zeros((labels.size, Num_classes), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    # Unlabeled dataset loading
    print('Loading images from "%s"' % unlabeled_dir)
    unlabeled_filenames = sorted(glob.glob(os.path.join(unlabeled_dir, '*')))
    print()

    # Checks
    if len(labeled_filenames) == 0:
        error('No input images found in ' + labeled_dir)
    if len(unlabeled_filenames) == 0:
        error('No input images found in ' + unlabeled_dir)

    # Checking to make sure dimensions are all good
    img = np.asarray(PIL.Image.open(labeled_filenames[0][0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')


    # NEED TO DEBUG!!!!!!!!!
    # Adding labeled data
    with TFRecordExporter(labeled_tfrecord_dir, len(labels)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(labels))
        #  Iterate through each class
        for i in range(Num_classes):
            # Go over each image in the class
            for idx in range(len(labeled_filenames[i])):
                img = np.asarray(PIL.Image.open(labeled_filenames[i][order[idx]]))
                if channels == 1:
                    img = img[np.newaxis, :, :]  # HW => CHW
                else:
                    img = img.transpose(2, 0, 1)  # HWC => CHW
                tfr.add_image(img)
        tfr.add_labels(onehot[order])

    print()

    #  Adding unlabeled data
    with TFRecordExporter(unlabeled_tfrecord_dir, len(unlabeled_filenames)) as tfr2:
        fake_labels = [Num_classes - 1] * len(unlabeled_filenames)
        fake_labels = np.array(fake_labels)
        fake_onehot = np.zeros((fake_labels.size, np.max(fake_labels) + 1), dtype=np.float32)
        fake_onehot[np.arange(fake_labels.size), fake_labels] = 1.0

        order = tfr2.choose_shuffled_order() if shuffle else np.arange(len(unlabeled_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(unlabeled_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose(2, 0, 1)  # HWC => CHW
            tfr2.add_image(img)
        tfr2.add_labels(fake_onehot[order])


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        error("Wrong amount of commands given!\nFormat: python3 dataset_tool.py <Labeled dir> <Unlabeled dir>\nEx) python3 dataset_tool.py /home/user/Desktop/SSL-PG-GAN/CatVDog/PetImages/Labeled/ /home/user/Desktop/SSL-PG-GAN/CatVDog/PetImages/Unlabeled/\n")

    if not os.path.isdir("Labeled"):
        os.mkdir("Labeled")
    if not os.path.isdir("Unlabeled"):
        os.mkdir("Unlabeled")

    args = sys.argv[1:]

    create_from_images("Labeled", "Unlabeled", args[0] + "/", args[1] + "/", False)

# ----------------------------------------------------------------------------



