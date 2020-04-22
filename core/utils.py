import numpy as np
import tensorflow as tf
import random
import cv2
import os
import glob
from core.config import cfg


def label_to_dict(path):
    """
    :param path: the path to label txt
    :return:
    """
    label_dict = {}
    texts = [label.strip() for label in open(path, mode="r").readlines()]
    for label in range(len(texts)):
        label_dict[texts[label]] = label
    return label_dict


def read_and_decode(filename):
    features = tf.io.parse_single_example(filename, features={
        "image/data_raw": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/path": tf.io.FixedLenFeature([], tf.string)
    })
    image_raw = features["image/data_raw"]
    image_path = features["image/path"]
    image = tf.image.decode_png(image_raw, channels=3)
    image = tf.image.resize(image, size=[cfg.IMG.SIZE, cfg.IMG.SIZE])
    label = features["image/class/label"]
    label = tf.one_hot(label, depth=cfg.DATA.CLASSES)
    image = tf.image.per_image_standardization(image)
    return image, label, image_path


def image_iterator(filenames, batchsize, epoch):
    """
    build a iterator of .Ani_train_tfrecord as being data input
    :param filenames: the Ani_train_tfrecord files as string list
    :param batchsize: batch
    :param epoch:epoch
    :return:
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode).shuffle(buffer_size=100).batch(batchsize).repeat(epoch)
    image_iter = dataset.make_one_shot_iterator()
    return image_iter.get_next()


def tfrecord_load(output_dir):
    """load Ani_train_tfrecord data"""
    print(output_dir)
    tfrecord_dir = output_dir + "/*"
    if len(tf.io.gfile.glob(tfrecord_dir)) == 0:
        raise FileExistsError("no Ani_train_tfrecord data found")
    return tf.io.gfile.glob(tfrecord_dir)


def remove_files(files_path):
    """
    remove the file which is not image. eg(dir, .mov) t
    :param files_path: list of string
    :return: files: list of string path
    """
    files = []
    for path in files_path:
        if (".jpg" in path) or (".png" in path):
            files.append(path)
    return files


def is_jpg(filename):
    return ".jpg" in filename


def jpg_to_png(image_data):
    """convert jpg to png with using tensorflow api"""
    sess = tf.Session()
    jpeg_data = tf.placeholder(tf.string)
    image = tf.image.decode_jpeg(jpeg_data, channels=3)
    jpeg_to_png = tf.image.encode_png(image)
    return sess.run(jpeg_to_png, feed_dict={jpeg_data: image_data})


def process_image(name, filename):
    """
    process a single image file
    :param filename: string, path to image
    :param name: string, whether a training or testing data
    :return:
            image_buffer: string , png encoding of RGB image
            height: image height in size
            width: image width
    """
    with tf.io.gfile.GFile(filename, "rb") as f:
        image_data = f.read()  # bytes type

    # if is_jpg(filename):
    #     print("convert jpg to png for %s" % filename)
    #     image_data = jpg_to_png(image_data)

    return image_data


def load_class(label_files):
    """
    :param label_files, the path to label txt
    """
    unique_labels = [l.strip() for l in open(label_files, "r").readlines()]
    return unique_labels


def find_img_file(data_dir, label_files):
    """
    build a lost of all images which are in files and labels in the dataset"
    :param data_dir: type string, path to the root directory of images
    :param label_files: string, path to the labels files
    :return:
            1. filenames: list of string, the image path
            2. texts: string of list, classification name(eg: zebra, elephant, pungin, lion, panda, others)
            3. labels: int list, 1,2,3,4,5,6 for respective animals files
    """
    print("target file location: %s" % data_dir)
    print("label file location %s" % label_files)
    unique_labels = load_class(label_files)

    labels, filenames, texts = [], [], []

    # label data with 0 as the start
    label_index = 0

    # populate the three lists
    for text in unique_labels:
        file_path = "%s/%s/*" % (data_dir, text)
        try:
            # read all image files, return string list
            matching_files = tf.io.gfile.glob(file_path)
        except Exception:
            print(file_path)
            continue

        # remove dir, video path
        matching_files = remove_files(matching_files)
        # start labeling per image
        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)
        # int label imcrement
        label_index += 1

    shuffle_index = list(range(len(filenames)))
    random.seed(41)
    random.shuffle(shuffle_index)

    # rearrange the index according to the shuffle index
    filenames = [filenames[i] for i in shuffle_index]
    texts = [texts[i] for i in shuffle_index]
    labels = [labels[i] for i in shuffle_index]

    print("Find %d the image files among %d labels in %s" % (len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels, unique_labels
