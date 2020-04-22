import tensorflow as tf
from pretrained_model.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from core import utils
from core.config import cfg
import os
import numpy as np
from test import Test
from sklearn.metrics import confusion_matrix

# from test import Test

slim = tf.contrib.slim


class Train(object):
    def __init__(self, pretrained):
        """
        :param pretrained: which pretrained model you want to train from scratch
        """
        self.image_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="image_input")
        self.label_input = tf.placeholder(tf.float32, shape=(None, cfg.DATA.CLASSES), name="label_input")

        if pretrained == "mobilenet_v1":
            self.logits, self.variables_to_restore = self.__mobilenet_v1_build()
            self.model_path = os.path.join(cfg.FINETUNE.MOBILENET_V1, "mobilenet_v1_1.0_224.ckpt")

            self.init_fn = slim.assign_from_checkpoint_fn(self.model_path, self.variables_to_restore)
            self.logits_variables1 = slim.get_variables('MobilenetV1/Logits/Conv2d_1c_1x1')
            self.logits_init1 = tf.variables_initializer(self.logits_variables1)
            self.logits_init_list = [self.logits_init1]

        self.global_step = tf.Variable(0.0, trainable=False, name="global_step")
        self.loss, self.train_step, self.learning_rate = self.loss_layer()
        self.score = tf.nn.softmax(logits=self.logits, name="score")
        self.global_step_update = tf.assign_add(self.global_step, 1.0)
        self.logits_one_dimension = tf.argmax(self.logits, axis=1, name="logits_one_dimension")
        self.pred = tf.equal(self.logits_one_dimension, tf.argmax(self.label_input, axis=1), name="pred")
        self.accuracy = tf.reduce_mean(tf.cast(self.pred, tf.float32), name="accuracy")

        self.saver = tf.train.Saver()

        os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.label_dict = utils.label_to_dict(path=cfg.DATA.LABEL_FILEPATH)
        self.label_dict = {value: key for key, value in self.label_dict.items()}

    def __mobilenet_v1_build(self):
        with slim.arg_scope(mobilenet_v1_arg_scope(is_training=True)):
            logits, _ = mobilenet_v1(self.image_input, is_training=True, num_classes=cfg.DATA.CLASSES)
            variables_to_restore = slim.get_variables_to_restore(exclude=['MobilenetV1/Logits/Conv2d_1c_1x1'])
        return logits, variables_to_restore

    def loss_layer(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_input, logits=self.logits))
        learning_rate = tf.train.exponential_decay(
            learning_rate=cfg.TRAIN.BASE_LR,
            global_step=self.global_step,
            decay_steps=int(cfg.IMG.TRAIN_NUMS / cfg.TRAIN.BATCH),
            decay_rate=0.96
        )
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return loss, train_step, learning_rate

    def train(self):
        self.sess.run(self.logits_init_list)
        self.init_fn(self.sess)
        self.sess.run(tf.global_variables_initializer())
        image_train_iter = utils.image_iterator(utils.tfrecord_load(cfg.DATA.TRAIN_OUTPUT_DIR),
                                                cfg.TRAIN.BATCH, cfg.TRAIN.EPOCH)

        while True:
            image, label, image_path = self.sess.run(image_train_iter)
            global_step_curr = self.sess.run(self.global_step_update)
            if global_step_curr % cfg.DISPLAY.TRAIN_DISPLAY == 0:
                train_accuracy, train_loss, lr, _, pred, score = self.sess.run(fetches=(self.accuracy,
                                                                                        self.loss,
                                                                                        self.learning_rate,
                                                                                        self.train_step,
                                                                                        self.pred,
                                                                                        self.score),
                                                                               feed_dict={
                                                                                   self.image_input: image,
                                                                                   self.label_input: label})
                # post-process the result and save logs
                self.saver.save(self.sess, save_path=cfg.LOGS.MODEL_SAVED_PATH, global_step=int(global_step_curr))
                print("train_acc:{:.6f}\t\ttrain_loss:{:.4f}\t\titer:{:.1f}\t\tlr:{:.8f}".format(train_accuracy,
                                                                                                 train_loss,
                                                                                                 global_step_curr,
                                                                                                 lr))

            if global_step_curr % cfg.DISPLAY.TEST_DISPLAY == 0 and global_step_curr != 0:
                path = (cfg.LOGS.MODEL_SAVED_PATH + "-%d") % global_step_curr
                test_accuracy, pred, score, test_img_path, label_test, logits = Test(path=path).test_start()
                self.__post_process(pred, test_img_path, score, labels=label_test, global_step=global_step_curr, logits=logits)
                print(("*" * 10 + "test_accuracy %.4f" + "*" * 10) % test_accuracy)

            if global_step_curr == 200000.0:
                break

    def __post_process(self, pred, image_path, score, labels, global_step, logits):
        """
        the prediction value fetched
        :param labels:
        :param score:
        :param image_path:
        :param pred: one-dimension
        :return:
        """
        false_id = np.where(pred == False)[0]
        mis_image_path_id = image_path[false_id].astype(str)
        score_max = np.max(score, axis=1)  # 64
        score_max_labels = np.argmax(score, axis=1)  # 64
        mis_label_scores = score_max[false_id]  # 错误的类别的得分值
        mis_labels = score_max_labels[false_id]  # 错误的预测成什么类别

        labels = np.argmax(labels, axis=1)  # 64
        correct_labels = labels[false_id]
        if not os.path.exists(cfg.LOGS.OUTPUT_DIR):
            os.mkdir(cfg.LOGS.OUTPUT_DIR)
        if not os.path.exists(cfg.LOGS.OUTPUT_SNAPSHOT_DIR):
            os.mkdir(cfg.LOGS.OUTPUT_SNAPSHOT_DIR)

        model_logs_txt = cfg.LOGS.MODEL_SAVED_PATH + "-{}.txt".format(int(global_step))
        if os.path.exists(model_logs_txt):
            os.remove(model_logs_txt)

        # save info into txt file
        for mis in range(len(mis_image_path_id)):
            path = mis_image_path_id[mis]
            mis_label_score = mis_label_scores[mis]
            mis_label = mis_labels[mis]
            correct_label = correct_labels[mis]

            with open(model_logs_txt, mode="a") as f:
                f.write(path + " " + str(mis_label_score) + " " + "class" + " " + str(mis_label) + " " +
                        str(self.label_dict[mis_label]) + " " + "GT" + " " + str(correct_label) + " " +
                        str(self.label_dict[correct_label]) + "\n")

        # rearrange the info to make it more formatted
        info_list = [info.strip() for info in open(model_logs_txt, "r").readlines()]
        gt_info = dict()
        for info in info_list:
            if info.split()[-2] not in gt_info:
                gt_info[info.split()[-2]] = []

            # append the info
            gt_info[info.split()[-2]].append(info)

        if os.path.exists(model_logs_txt):
            os.remove(model_logs_txt)

        # create confusion matrix
        confuse_matrix = confusion_matrix(y_pred=logits, y_true=labels)
        np.savetxt(model_logs_txt[0:-4] + "-confusion.txt", confuse_matrix, fmt="%s", delimiter=",")

        for info_key in gt_info:
            f = open(model_logs_txt, mode="a")
            for i in gt_info[info_key]:
                f.write(i)
                f.write("\n")
            f.write("\n\n\n")
            f.close()

if __name__ == '__main__':
    # # gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    # # sess = tf.Session(config=gpu_config)
    # image_inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="image_input")
    # label_input = tf.placeholder(tf.float32, shape=(None, cfg.DATA.CLASSES), name="label_input")
    #
    # with slim.arg_scope(mobilenet_v1_arg_scope(is_training=True)):
    #     logits, _ = mobilenet_v1(image_inputs, is_training=True, num_classes=cfg.DATA.CLASSES)
    #
    # variables_to_restore = slim.get_variables_to_restore(exclude=['MobilenetV1/Logits/Conv2d_1c_1x1'])
    # ckpt = os.path.join(cfg.FINETUNE.MOBILENET_V1, "mobilenet_v1_1.0_224.ckpt")
    # init_fn = slim.assign_from_checkpoint_fn(ckpt, variables_to_restore)
    #
    # logits_variables1 = slim.get_variables('MobilenetV1/Logits/Conv2d_1c_1x1')
    # logits_init1 = tf.variables_initializer(logits_variables1)
    # logits_init_list = [logits_init1]
    # global_step = tf.Variable(0.0, trainable=False, name="global_step")
    #
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    # learning_rate = tf.train.exponential_decay(
    #     learning_rate=cfg.TRAIN.BASE_LR,
    #     global_step=global_step,
    #     decay_steps=int(cfg.IMG.NUMS / cfg.TRAIN.BATCH),
    #     decay_rate=0.96
    # )
    # global_step_update = tf.assign_add(global_step, 1.0)
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #
    # pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    # accuracy = tf.reduce_mean(tf.cast(pred, tf.float32), name="accuracy")
    #
    # # save the model
    # saver = tf.train.Saver(max_to_keep=1)
    # # initialize and load the data
    # sess.run(logits_init_list)
    # init_fn(sess)
    # sess.run(tf.global_variables_initializer())
    # image_train_iter = utils.image_iterator(utils.tfrecord_load(cfg.DATA.TRAIN_OUTPUT_DIR),
    #                                         cfg.TRAIN.BATCH, cfg.TRAIN.EPOCH)
    #
    # while True:
    #     image, label = sess.run(image_train_iter)
    #     global_step_curr = sess.run(global_step_update)
    #
    #     if global_step_curr % cfg.DISPLAY.TRAIN_DISPLAY == 0:
    #         train_accuracy, train_loss, lr, _= sess.run(fetches=(accuracy, loss, learning_rate, train_step), feed_dict={inputs: image,
    #                                                                                                    labels: label})
    #         print("train_acc:{:.6f}\t\ttrain_loss:{:.4f}\t\titer:{:.1f}\t\tlr:{:.8f}".format(train_accuracy, train_loss,
    #                                                                                          global_step_curr,
    #                                                                                          lr))
    #         saver.save(sess, save_path=cfg.LOGS.MODEL_SAVED_PATH, global_step=int(global_step_curr))
    #


    Train(pretrained="mobilenet_v1").train()
