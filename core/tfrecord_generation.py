import numpy as np
import tensorflow as tf
import threading
import os
import sys
import cv2
from datetime import datetime
from core import utils
from core.config import cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class TfGen(object):
    """
    the class of building tfrecord data
    """

    def __init__(self, name, img_dir, output_dir, label_txt_path, shard_nums, thread_nums):
        """
        :param name: train or test
        :param img_dir: path to the image
        :param output_dir: output dir of tfrecord data
        :param label_txt_path: the path to label txt
        :param shard_nums: the number of tfrecords to be saved
        :param thread_nums: the number of threads to be launched
        """
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.label_txt_path = label_txt_path
        self.shard_nums = shard_nums
        self.thread_nums = thread_nums

        self.filenames, self.texts, self.labels, self.unique_labels = utils.find_img_file(img_dir, label_files=self.label_txt_path)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.image_process_thread(name=name)

    def image_process_thread(self, name):
        self.__image_process_thread(name,
                                    filenames=self.filenames,
                                    cls_names=self.texts,
                                    labels=self.labels,
                                    output_dir=self.output_dir,
                                    num_threads=self.thread_nums,
                                    shards_nums=self.shard_nums)

    def __convert_to_example(self, image_buff, label, cls_name, img_path):
        """build example proto"""
        features = tf.train.Features(feature={
            "image/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "image/class/cls_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(cls_name)])),
            "image/data_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_buff])),
            "image/path": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_path)])
            )
        })
        example = tf.train.Example(features=features)
        return example

    def __proces_image_shards(self, thread_index, ranges, name, filenames, cls_names, labels, output_dir, shards_num):
        """
        process and save the list if images in 1 thread
        :param thread_index: int, deal with images with index:[0, len(ranges))
        :param ranges:list of pairs of integers specifying ranges of each shard in parallel
        :param name:string, unique identifier specifying the dataset
        :param filenames:ist of strings, each string is the path to an image
        :param cls_names:list of strings, classfication name
        :param labels:int-type of labels
        :return:
        """
        num_threads = len(ranges)
        assert not shards_num % num_threads
        num_shards_per_batch = int(shards_num / num_threads)

        # [1835, 3670]
        # [1835, 2752, 3670]
        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(
            np.int)
        num_files_in_threads = ranges[thread_index][1] - ranges[thread_index][0]
        counter = 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a sharded version of file name, eg: "train-00002-of-00010"
        for s in range(num_shards_per_batch):
            shard = thread_index * num_shards_per_batch + s
            output_filename = "%s-%.5d-of-%.5d.tfrecord" % (name, shard, shards_num)

            output_file = os.path.join(output_dir, output_filename)
            writer = tf.io.TFRecordWriter(output_file)

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                filename = filenames[i]
                label = labels[i]
                cls_name = cls_names[i]

                image_buffer = utils.process_image(name, filename)
                example = self.__convert_to_example(image_buff=image_buffer, label=label, cls_name=cls_name, img_path=filename)
                writer.write(example.SerializeToString())

                shard_counter += 1
                counter += 1
                if not counter % 5000:
                    print("%s [thread %d]: Process %d of %d images in thread batch" %
                          (datetime.now(), thread_index, counter, num_files_in_threads))

            writer.close()
            print("%s [thread %d]: wrote %d images to %s" %
                  (datetime.now(), thread_index, shard_counter, output_file))

            sys.stdout.flush()

        print("%s [thread %d]: wrote %d images to %d shards" %
              (datetime.now(), thread_index, counter, num_files_in_threads))

        sys.stdout.flush()

    def __image_process_thread(self, name, filenames, cls_names, labels, output_dir, num_threads=4, shards_nums=2):
        """
        image preparation and read by thread
        :param name: string, unique identifier specifying the dataset
        :param filenames:  list of strings, each string is the path to an image
        :param cls_names: list of strings, classfication name
        :param labels: int-type of labels
        :return:
        """
        assert len(filenames) == len(cls_names)
        assert len(filenames) == len(labels)

        # Break all images into shards with [ranges[i][0], ranges[i][1]]
        spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
        # ranges will finally are the result like [[0, 1835], [1835 3670]]
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        print("lauching the %d number of threads for image amount:%s" % (num_threads, ranges))

        sys.stdout.flush()
        coord = tf.train.Coordinator()
        threads = []
        for thread_index in range(len(ranges)):
            args = (thread_index, ranges, name, filenames, cls_names, labels, output_dir, shards_nums)
            t = threading.Thread(target=self.__proces_image_shards, args=args)
            t.start()
            threads.append(t)

        coord.join(threads)
        print("%s: Finished convert all %d images in dataset" % (datetime.now(), len(filenames)))

    def img_info(self, img_info_path):
        f = open(img_info_path, mode="w")
        f.write("img_nums: {}\nnum_classes: {}".format(len(self.filenames), len(self.unique_labels)))
        f.close()