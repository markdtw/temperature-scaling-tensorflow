import os
import pdb
import pickle
import numpy as np
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

def _read_data(files):
    """Reads CIFAR-10 format data. Always returns NHWC format.

    Returns:
        images: np tensor of size [N, H, W, C]
        labels: np tensor of size [N]
    """
    images, labels = [], []
    for file_name in files:
        print (file_name)
        with open(file_name, 'rb') as finp:
            data = pickle.load(finp, encoding='bytes')
            batch_images = data[b'data'].astype(np.float32) / 255.0
            if 'cifar-100' in file_name:
                batch_labels = np.array(data[b'fine_labels'], dtype=np.int32)
            else:
                batch_labels = np.array(data[b'labels'], dtype=np.int32)
            images.append(batch_images)
            labels.append(batch_labels)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    images = np.reshape(images, [-1, 3, WIDTH, HEIGHT])
    images = np.transpose(images, [0, 2, 3, 1])

    return images, labels

def read_data(data_path, num_valids=5000):
    print("Reading data from {}".format(data_path))

    images, labels = {}, {}

    if 'cifar-100' in data_path:
        train_files = [
            os.path.join(data_path, 'train')
        ]
        test_file = [
            os.path.join(data_path, 'test')
        ]
    else:
        train_files = [
            os.path.join(data_path, 'data_batch_1'),
            os.path.join(data_path, 'data_batch_2'),
            os.path.join(data_path, 'data_batch_3'),
            os.path.join(data_path, 'data_batch_4'),
            os.path.join(data_path, 'data_batch_5'),
        ]
        test_file = [
            os.path.join(data_path, 'test_batch'),
        ]
    images['train'], labels['train'] = _read_data(train_files)

    if num_valids:
        images['valid'] = images['train'][-num_valids:]
        labels['valid'] = labels['train'][-num_valids:]

        images['train'] = images['train'][:-num_valids]
        labels['train'] = labels['train'][:-num_valids]
    else:
        images['valid'], labels['valid'] = None, None

    images['test'], labels['test'] = _read_data(test_file)

    print("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images['train'], axis=(0, 1, 2), keepdims=True)
    std = np.std(images['train'], axis=(0, 1, 2), keepdims=True)

    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images['train'] = (images['train'] - mean) / std
    if num_valids:
        images['valid'] = (images['valid'] - mean) / std
    images['test'] = (images['test'] - mean) / std

    return images, labels


class CifarDataSet:

    def __init__(self, batch_size, data_dir, eval_batch_size=100):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.images_np, self.labels_np = read_data(data_dir)
        self.validation = tf.placeholder(tf.bool)

    def make_batch_train(self):
        # Extract
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.constant(self.images_np['train']), tf.constant(self.labels_np['train'])))

        # Transform
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(10000))
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self._pre_process, self.batch_size, num_parallel_batches=4))
        dataset = dataset.prefetch(self.batch_size)

        # Load
        iterator = dataset.make_one_shot_iterator()
        self.images_train, self.labels_train = iterator.get_next()

    def make_batch_valid_or_test(self):
        # Extract
        images, labels = tf.cond(self.validation,
            lambda: (self.images_np['valid'], self.labels_np['valid']),
            lambda: (self.images_np['test'], self.labels_np['test']))
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Transform
        dataset = dataset.batch(self.eval_batch_size)
        dataset = dataset.prefetch(self.eval_batch_size)

        # Load
        self.iterator_vt = dataset.make_initializable_iterator()
        self.images_vt, self.labels_vt = self.iterator_vt.get_next()

    def _pre_process(self, image, label):
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
        image = tf.image.random_flip_left_right(image)
        return image, label
