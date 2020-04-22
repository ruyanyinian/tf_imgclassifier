import tensorflow as tf
from core.config import cfg
from core import utils
import os


class Test(object):
    def __init__(self, path):
        """
        path: the path to saved model without .meta
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"
        g = tf.Graph()
        with g.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                                                                   per_process_gpu_memory_fraction=0.9)))

            self.graph = tf.get_default_graph()
            self.__restore = tf.train.import_meta_graph(meta_graph_or_file=path + ".meta").restore(save_path=path,
                                                                                                   sess=self.sess)

            # get orginal nodes
            self.__test_image = self.graph.get_tensor_by_name("image_input:0")
            self.__test_label = self.graph.get_tensor_by_name("label_input:0")
            self.__logits = self.graph.get_tensor_by_name('logits_one_dimension:0')
            self.__test_accuracy = self.graph.get_tensor_by_name("accuracy:0")
            self.__pred = self.graph.get_tensor_by_name("pred:0")
            self.__score = self.graph.get_tensor_by_name("score:0")

            # get data from Ani_train_tfrecord
            self.__test_tfrecord = utils.tfrecord_load(cfg.DATA.TEST_OUTPUT_DIR)
            self.__image_test_iter = utils.image_iterator(self.__test_tfrecord, cfg.TEST.BATCH, cfg.TEST.EPOCH)

    def test_start(self):
        image_test, label_test, test_path = self.sess.run(self.__image_test_iter)
        test_accuracy_val, pred, score, logits = self.sess.run((self.__test_accuracy, self.__pred, self.__score, self.__logits),
                                          feed_dict={self.__test_image: image_test,
                                                     self.__test_label: label_test})
        return test_accuracy_val, pred, score, test_path, label_test, logits


if __name__ == '__main__':
    while True:
        path = "./checkpoint/AnimalDector-211"
        test_accuracy = Test(path=path).test_start()
        print(("*" * 10 + "test_accuracy %.4f" + "*" * 10) % test_accuracy)
